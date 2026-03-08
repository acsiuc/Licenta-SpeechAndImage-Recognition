import torch
import torchaudio
import os
from torch.utils.data import DataLoader
from dataset import MavCelebDataset # Your existing dataset class
from models import FaceEncoder, VoiceEncoder
torchaudio.set_audio_backend("soundfile")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mavceleb_v1_test"
OUTPUT_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings" # Where to save vectors

def extract():
    # 1. Setup Models (Pretrained)
    print("Loading models for extraction...")
    face_net = FaceEncoder().to(DEVICE)
    voice_net = VoiceEncoder().to(DEVICE)
    face_net.eval()
    voice_net.eval()

    # 2. Setup Data
    dataset = MavCelebDataset(DATA_DIR)
    # Important: No shuffle, we just want to process everything once
    loader = DataLoader(dataset, batch_size=1, shuffle=False) 

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Starting extraction for {len(dataset)} samples...")

    with torch.no_grad():
        for i, (face_img, voice_spec, label) in enumerate(loader):
            face_img, voice_spec = face_img.to(DEVICE), voice_spec.to(DEVICE)

            # Get Embeddings (Vectors)
            face_emb = face_net(face_img)
            voice_emb = voice_net(voice_spec.squeeze(1))

            # Save them to disk
            # Structure: data_ID.pt
            save_path = os.path.join(OUTPUT_DIR, f"sample_{i}.pt")
            torch.save({
                'face_emb': face_emb.cpu(),
                'voice_emb': voice_emb.cpu(),
                'label': label.cpu()
            }, save_path)

            if i % 100 == 0:
                print(f"Processed {i} samples...")

if __name__ == "__main__":
    extract()