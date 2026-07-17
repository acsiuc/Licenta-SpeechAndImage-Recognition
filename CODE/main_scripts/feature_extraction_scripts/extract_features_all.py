import torch
import torchaudio
import os
import glob
import random
import torch.nn.functional as F
import cv2
from models import FaceEncoder, VoiceEncoder

try:
    torchaudio.set_audio_backend("soundfile")  
except:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# face embeddings never actually get moved to DEVICE, only voice does
# ArcFace runs through insightface's own app.get() call, not a normal torch module, so .to() doesnt apply to it

# All three datasets with their available languages
DATASETS = [
    {
        "path": r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mavceleb_v1_train",
        "languages": ["English"]  # Urdu is deliberately left out here, its held out entirely for test_urdu.py later, not a bug
    },
    {
        "path": r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\v2",
        "languages": ["English", "Hindi"]
    },
    {
        "path": r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mav_celeb_v3_train",
        "languages": ["English", "German"]
    },
]

OUTPUT_DIR  = r"C:\Users\Axiuc\Downloads\mavceleb_all_embeddings"
MULTIPLIER  = 500  # pairs per identity per language combination, with replacement

def get_files(root, identity, language, ext):
    pattern = os.path.join(root, identity, language, "**", f"*.{ext}")
    return glob.glob(pattern, recursive=True)

def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    face_net  = FaceEncoder().eval()
    voice_net = VoiceEncoder().eval()

    # Build global identity map across all datasets
    # key: (dataset_idx, identity_folder) → label
    identity_map = {}
    label_counter = 0

    for ds_idx, ds in enumerate(DATASETS):
        face_root = os.path.join(ds["path"], "faces")
        if not os.path.exists(face_root):
            print(f"Skipping {ds['path']} — faces folder not found")
            continue
        identities = sorted([d for d in os.listdir(face_root) if d.startswith("id")])
        for identity in identities:
            key = (ds_idx, identity)
            if key not in identity_map:
                # this check never actually triggers false, keys are already unique tuples by construction
                # kept as a safety net, not a real branch
                identity_map[key] = label_counter
                label_counter += 1

    print(f"Total unique identities: {label_counter}")

    sample_idx = 0

    with torch.no_grad():
        for (ds_idx, identity), label in identity_map.items():
            ds = DATASETS[ds_idx]
            face_root  = os.path.join(ds["path"], "faces")
            voice_root = os.path.join(ds["path"], "voices")
            languages  = ds["languages"]

            # collect all face and voice files per language
            # this is what makes cross-language pairing possible below
            face_files  = {}
            voice_files = {}
            for lang in languages:
                ff = get_files(face_root, identity, lang, "jpg")
                vf = get_files(voice_root, identity, lang, "wav")
                if ff: face_files[lang]  = ff
                if vf: voice_files[lang] = vf

            if not face_files or not voice_files:
                # this is where the "empty folder" identities get dropped
                # same issue that later shows up as 9 dropped folders in test_in_the_wild.py
                print(f"Skipping {identity} in {ds['path']} — missing data")
                continue

            face_langs  = list(face_files.keys())
            voice_langs = list(voice_files.keys())

            for _ in range(MULTIPLIER):
                # randomly pick face language and voice language INDEPENDENTLY
                # sometimes same language, sometimes cross-language on purpose
                # this is the actual mechanism forcing the model to learn identity, not language
                f_lang = random.choice(face_langs)
                v_lang = random.choice(voice_langs)

                face_path  = random.choice(face_files[f_lang])
                voice_path = random.choice(voice_files[v_lang])

                # face embedding
                # face embedding — ArcFace needs BGR numpy image
                try:
                    img_bgr  = cv2.imread(face_path)
                    if img_bgr is None:
                        continue
                    faces    = face_net.app.get(img_bgr)
                    if len(faces) == 0:
                        continue
                    face_emb = torch.tensor(faces[0].embedding, dtype=torch.float32).unsqueeze(0)
                    face_emb = F.normalize(face_emb, p=2, dim=1)
                except Exception as e:
                
                    continue

                # voice embedding
                try:
                    waveform, sr = torchaudio.load(voice_path)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                    target_len = 16000 * 3
                    if waveform.shape[1] < target_len:
                        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
                    else:
                        # center crop, deterministic, NOT random like MavCelebDataset.__getitem__ does
                        # this script has zero augmentation, one fixed embedding per file per draw
                        start    = (waveform.shape[1] - target_len) // 2
                        waveform = waveform[:, start: start + target_len]
                    waveform_input = waveform.squeeze(0).unsqueeze(0).to(DEVICE)
                    voice_emb      = voice_net(waveform_input)
                except Exception as e:
                    continue

                save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:06d}.pt")
                torch.save({
                    'face_emb':  face_emb.cpu(),   # moved to cpu so the .pt files are loadable on any machine later, gpu or not
                    'voice_emb': voice_emb.cpu(),
                    'label':     torch.tensor(label, dtype=torch.long),
                    'face_lang': f_lang,
                    'voice_lang': v_lang,
                }, save_path)
                sample_idx += 1

            if label % 20 == 0:
                # prints every 20th IDENTITY, not every 20th sample
                # so the sample count jumps are uneven since drop rate varies per identity
                print(f"  Processed {label}/{label_counter} identities, {sample_idx} samples so far...")

    print(f"\nDone. {sample_idx} total samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    random.seed(42)
    extract()