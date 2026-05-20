import torch
import torchaudio
import os
from models import FaceEncoder, VoiceEncoder
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mav_celeb_v1_test"

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_split(pairs_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    face_net  = FaceEncoder().to(DEVICE).eval()
    voice_net = VoiceEncoder().to(DEVICE).eval()

    with open(pairs_file, 'r') as f:
        lines = f.read().strip().split('\n')

    print(f"Extracting {len(lines)} pairs from {os.path.basename(pairs_file)}...")

    with torch.no_grad():
        for i, line in enumerate(lines):
            parts = line.strip().split()
            pair_id    = parts[0]
            voice_path = os.path.join(TEST_DIR, parts[1])
            face_path  = os.path.join(TEST_DIR, parts[2])

            # face
            img         = Image.open(face_path).convert("RGB")
            face_tensor = face_transform(img).unsqueeze(0).to(DEVICE)
            face_emb    = face_net(face_tensor)

            # voice
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

            save_path = os.path.join(output_dir, f"{pair_id}.pt")
            torch.save({
                'face_emb':  face_emb.cpu(),
                'voice_emb': voice_emb.cpu(),
                'pair_id':   pair_id,
                'label':     torch.tensor(i, dtype=torch.long)
            }, save_path)

            if i % 100 == 0:
                print(f"  {i}/{len(lines)} done...")

    print(f"Done. {len(lines)} pairs saved to {output_dir}")

if __name__ == "__main__":
    extract_split(
        pairs_file  = os.path.join(TEST_DIR, "English_test.txt"),
        output_dir  = r"C:\Users\Axiuc\Downloads\mavceleb_test_english"
    )
    extract_split(
        pairs_file  = os.path.join(TEST_DIR, "Urdu_test.txt"),
        output_dir  = r"C:\Users\Axiuc\Downloads\mavceleb_test_urdu"
    )