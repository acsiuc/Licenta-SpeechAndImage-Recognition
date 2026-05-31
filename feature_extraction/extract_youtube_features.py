import torch
import torchaudio
import os
import glob
from models import FaceEncoder, VoiceEncoder
import cv2
import torch.nn.functional as F 

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\build_youtube_corpus"
OUTPUT_DIR = r"C:\Users\Axiuc\Downloads\youtube_embeddings"

FACE_DIR  = os.path.join(CORPUS_DIR, "faces", "English")
VOICE_DIR = os.path.join(CORPUS_DIR, "voices", "English")

def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    face_net  = FaceEncoder().eval()
    voice_net = VoiceEncoder().eval()

    # get unique identity names from voice files
    voice_files = glob.glob(os.path.join(VOICE_DIR, "*.wav"))
    identities  = sorted([os.path.splitext(os.path.basename(f))[0] for f in voice_files])
    id_to_label = {name: i for i, name in enumerate(identities)}

    print(f"Found {len(identities)} identities: {identities}")

    sample_idx = 0
    with torch.no_grad():
        for identity in identities:
            label = id_to_label[identity]

            # load and average all face images for this identity
            face_paths = sorted(glob.glob(os.path.join(FACE_DIR, f"{identity}_*.jpg")))
            if not face_paths:
                print(f"WARNING: no faces found for {identity}")
                continue

            # load voice
            voice_path = os.path.join(VOICE_DIR, f"{identity}.wav")
            waveform, sr = torchaudio.load(voice_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            # take 3 seconds from middle
            target_len = 16000 * 3
            if waveform.shape[1] < target_len:
                waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
            else:
                start = (waveform.shape[1] - target_len) // 2
                waveform = waveform[:, start: start + target_len]

            waveform_input = waveform.squeeze(0).unsqueeze(0).to(DEVICE)
            voice_emb = voice_net(waveform_input)

            # save one sample per face image
            for face_path in face_paths:
                img_bgr = cv2.imread(face_path)
                if img_bgr is None:
                    continue
                faces = face_net.app.get(img_bgr)
                if len(faces) == 0:
                    continue
                face_emb = torch.tensor(faces[0].embedding, dtype=torch.float32).unsqueeze(0)
                face_emb = F.normalize(face_emb, p=2, dim=1)

                save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:04d}.pt")
                torch.save({
                    'face_emb':  face_emb.cpu(),
                    'voice_emb': voice_emb.cpu(),
                    'label':     torch.tensor(label, dtype=torch.long)
                }, save_path)
                sample_idx += 1
                print(f"Saved {identity} face {sample_idx}")

    print(f"Done. {sample_idx} samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract()