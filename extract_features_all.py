import torch
import torchaudio
import os
import glob
import random
import torch.nn.functional as F
from models import FaceEncoder, VoiceEncoder
from torchvision import transforms
from PIL import Image

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All three datasets with their available languages
DATASETS = [
    {
        "path": r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mavceleb_v1_train",
        "languages": ["English", "Urdu"]
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
MULTIPLIER  = 500  # pairs per identity per language combination
FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_files(root, identity, language, ext):
    pattern = os.path.join(root, identity, language, "**", f"*.{ext}")
    return glob.glob(pattern, recursive=True)

def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    face_net  = FaceEncoder().to(DEVICE).eval()
    voice_net = VoiceEncoder().to(DEVICE).eval()

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
            face_files  = {}
            voice_files = {}
            for lang in languages:
                ff = get_files(face_root, identity, lang, "jpg")
                vf = get_files(voice_root, identity, lang, "wav")
                if ff: face_files[lang]  = ff
                if vf: voice_files[lang] = vf

            if not face_files or not voice_files:
                print(f"Skipping {identity} in {ds['path']} — missing data")
                continue

            face_langs  = list(face_files.keys())
            voice_langs = list(voice_files.keys())

            for _ in range(MULTIPLIER):
                # randomly pick face language and voice language
                # sometimes same language, sometimes cross-language
                f_lang = random.choice(face_langs)
                v_lang = random.choice(voice_langs)

                face_path  = random.choice(face_files[f_lang])
                voice_path = random.choice(voice_files[v_lang])

                # face embedding
                try:
                    img         = Image.open(face_path).convert("RGB")
                    face_tensor = FACE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
                    face_emb    = face_net(face_tensor)
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
                        start    = (waveform.shape[1] - target_len) // 2
                        waveform = waveform[:, start: start + target_len]
                    waveform_input = waveform.squeeze(0).unsqueeze(0).to(DEVICE)
                    voice_emb      = voice_net(waveform_input)
                except Exception as e:
                    continue

                save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:06d}.pt")
                torch.save({
                    'face_emb':  face_emb.cpu(),
                    'voice_emb': voice_emb.cpu(),
                    'label':     torch.tensor(label, dtype=torch.long),
                    'face_lang': f_lang,
                    'voice_lang': v_lang,
                }, save_path)
                sample_idx += 1

            if label % 20 == 0:
                print(f"  Processed {label}/{label_counter} identities, {sample_idx} samples so far...")

    print(f"\nDone. {sample_idx} total samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    random.seed(42)
    extract()