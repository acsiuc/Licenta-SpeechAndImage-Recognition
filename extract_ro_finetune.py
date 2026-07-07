import os, sys, glob, random
import torch
import torchaudio
import torch.nn.functional as F
import cv2

sys.path.insert(0, '/content/drive/MyDrive/Licenta_Colab/code')
os.chdir('/content/drive/MyDrive/Licenta_Colab/code')

from models import FaceEncoder, VoiceEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RO_ROOT = "/content/drive/MyDrive/Licenta_Colab/ro_facevoice"
FACE_ROOT = os.path.join(RO_ROOT, "faces", "Romanian")
VOICE_ROOT = os.path.join(RO_ROOT, "voices", "Romanian")

OUTPUT_DIR = "/content/drive/MyDrive/Licenta_Colab/embeddings"  # same folder as your MAVCeleb embeddings
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 33 training identities (10 held out for testing: ID002, ID011, ID012, ID013, ID027, ID029, ID035, ID036, ID037, ID045)
HELD_OUT = {"ID002", "ID011", "ID012", "ID013", "ID027", "ID029", "ID035", "ID036", "ID037", "ID045"}
ALL_VALID = [f"ID{str(i).zfill(3)}" for i in range(1, 51)
             if f"ID{str(i).zfill(3)}" not in {"ID003", "ID014", "ID016", "ID019", "ID020", "ID022", "ID046"}]  # exclude missing-face identities
TRAIN_IDS = [i for i in ALL_VALID if i not in HELD_OUT]

print(f"Training identities: {len(TRAIN_IDS)}")

# labels continue from 198 (the existing MAVCeleb identity count)
LABEL_START = 198
id_to_label = {ident: LABEL_START + i for i, ident in enumerate(TRAIN_IDS)}

CROPS_PER_IDENTITY = 3   # non-overlapping 3s segments sliced from voice_full.wav
CROP_SECONDS = 3.0
SAMPLE_RATE = 16000


def get_audio_crops(full_wav_path, n_crops, crop_seconds, sr):
    """Slice n_crops non-overlapping segments from the full audio track."""
    waveform, orig_sr = torchaudio.load(full_wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)

    total_len = waveform.shape[1]
    crop_len = int(crop_seconds * sr)

    if total_len < crop_len:
        # too short — just pad and use once
        padded = F.pad(waveform, (0, crop_len - total_len))
        return [padded]

    max_start = total_len - crop_len
    if max_start == 0:
        return [waveform[:, :crop_len]]

    # evenly spaced, non-overlapping-ish starting points
    starts = [int(max_start * i / max(1, n_crops - 1)) if n_crops > 1 else max_start // 2
              for i in range(n_crops)]
    crops = [waveform[:, s:s + crop_len] for s in starts]
    return crops


face_net = FaceEncoder().eval()
voice_net = VoiceEncoder().eval()

sample_idx = 0

with torch.no_grad():
    for ident in TRAIN_IDS:
        label = id_to_label[ident]

        face_paths = sorted(glob.glob(os.path.join(FACE_ROOT, ident, "*.jpg")))
        full_voice_path = os.path.join(VOICE_ROOT, ident, "voice_full.wav")

        if not face_paths or not os.path.exists(full_voice_path):
            print(f"Skipping {ident} — missing face or voice_full.wav")
            continue

        # get multiple voice crops for this identity
        try:
            voice_crops = get_audio_crops(full_voice_path, CROPS_PER_IDENTITY, CROP_SECONDS, SAMPLE_RATE)
        except Exception as e:
            print(f"Skipping {ident} — voice crop error: {e}")
            continue

        voice_embs = []
        for crop in voice_crops:
            waveform_input = crop.to(DEVICE)
            try:
                emb = voice_net(waveform_input)
                voice_embs.append(emb.cpu())
            except Exception as e:
                print(f"  voice embed failed for {ident}: {e}")

        if not voice_embs:
            continue

        # pair each available face image with a randomly chosen voice crop embedding
        for face_path in face_paths:
            img_bgr = cv2.imread(face_path)
            if img_bgr is None:
                continue
            faces = face_net.app.get(img_bgr)
            if len(faces) == 0:
                continue
            face_emb = torch.tensor(faces[0].embedding, dtype=torch.float32).unsqueeze(0)
            face_emb = F.normalize(face_emb, p=2, dim=1)

            voice_emb = random.choice(voice_embs)

            save_path = os.path.join(OUTPUT_DIR, f"ro_sample_{sample_idx:05d}.pt")
            torch.save({
                'face_emb': face_emb.cpu(),
                'voice_emb': voice_emb,
                'label': torch.tensor(label, dtype=torch.long),
                'face_lang': 'Romanian',
                'voice_lang': 'Romanian',
            }, save_path)
            sample_idx += 1

        print(f"{ident}: {len(face_paths)} faces x {len(voice_embs)} voice crops -> samples saved")

print(f"\nDone. {sample_idx} Romanian training samples saved to {OUTPUT_DIR}")