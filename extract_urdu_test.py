import torch
import torchaudio
import os
import glob
import random
import torch.nn.functional as F
from models import FaceEncoder, VoiceEncoder, ModalityTranslator
from torchvision import transforms
from PIL import Image

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mavceleb_v1_train"
OUT_DIR  = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract():
    os.makedirs(OUT_DIR, exist_ok=True)

    face_net  = FaceEncoder().to(DEVICE).eval()
    voice_net = VoiceEncoder().to(DEVICE).eval()

    face_root  = os.path.join(DATA_DIR, "faces")
    voice_root = os.path.join(DATA_DIR, "voices")

    identities = sorted([d for d in os.listdir(face_root) if d.startswith('id')])
    print(f"Found {len(identities)} identities")

    sample_idx = 0

    with torch.no_grad():
        for identity in identities:
            label = int(identity.replace('id', '')) - 1

            # get all Urdu face images for this identity
            face_paths = glob.glob(
                os.path.join(face_root, identity, "Urdu", "**", "*.jpg"),
                recursive=True
            )
            # get all Urdu voice files for this identity
            voice_paths = glob.glob(
                os.path.join(voice_root, identity, "Urdu", "**", "*.wav"),
                recursive=True
            )

            if not face_paths or not voice_paths:
                print(f"Skipping {identity} — no Urdu data")
                continue

            # pick one random face and one random voice
            face_path  = random.choice(face_paths)
            voice_path = random.choice(voice_paths)

            # face embedding
            img         = Image.open(face_path).convert("RGB")
            face_tensor = face_transform(img).unsqueeze(0).to(DEVICE)
            face_emb    = face_net(face_tensor)

            # voice embedding
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

            save_path = os.path.join(OUT_DIR, f"sample_{sample_idx:04d}.pt")
            torch.save({
                'face_emb':  face_emb.cpu(),
                'voice_emb': voice_emb.cpu(),
                'label':     torch.tensor(label, dtype=torch.long)
            }, save_path)
            sample_idx += 1
            print(f"Saved {identity} (label {label})")

    print(f"\nDone. {sample_idx} Urdu test samples saved.")

if __name__ == "__main__":
    random.seed(42)
    extract()