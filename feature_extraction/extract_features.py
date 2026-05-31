import torch
import torchaudio
import os
from torch.utils.data import DataLoader
from dataset import MavCelebDataset 
from models import FaceEncoder, VoiceEncoder
torchaudio.set_audio_backend("soundfile") 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
DATA_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mavceleb_v1_test" # path toraw images and audio
OUTPUT_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings" # path to save the .pt vectors

def extract():
    print("Loading models for extraction...")
    face_net = FaceEncoder().to(DEVICE) # spawn the face base model
    voice_net = VoiceEncoder().to(DEVICE) # spawn the voice base model
    face_net.eval() 
    voice_net.eval() 

    
    dataset = MavCelebDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR) 

    print(f"Starting extraction for {len(dataset)} samples...")

    with torch.no_grad():
        for i, (face_img, waveform, voice_spec, label) in enumerate(loader):
            face_img = face_img.to(DEVICE)
            waveform  = waveform.to(DEVICE)

            face_emb  = face_net(face_img)
            voice_emb = voice_net(waveform)  # raw waveform, not spectrogram

            for b in range(face_emb.shape[0]):
                sample_idx = i * loader.batch_size + b
                save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}.pt")
                torch.save({
                    'face_emb':  face_emb[b].unsqueeze(0).cpu(),
                    'voice_emb': voice_emb[b].unsqueeze(0).cpu(),
                    'label':     label[b].cpu()
                }, save_path)

            if i % 100 == 0:
                print(f"Processed batch {i} ({i * loader.batch_size} samples)...")

if __name__ == "__main__":
    extract()