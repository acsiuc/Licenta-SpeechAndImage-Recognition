import torch
import torchaudio
import os
from torch.utils.data import DataLoader
from dataset import MavCelebDataset 
from models import FaceEncoder, VoiceEncoder
torchaudio.set_audio_backend("soundfile") # sets audio backend so it doesn't crash on windows


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if we have one
DATA_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mavceleb_v1_test" # path to heavy raw images and audio
OUTPUT_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings" # path to save the tiny pre-baked .pt vectors

def extract():
    print("Loading models for extraction...")
    face_net = FaceEncoder().to(DEVICE) # spawn the face base model
    voice_net = VoiceEncoder().to(DEVICE) # spawn the voice base model
    face_net.eval() # lock face model in exam mode
    voice_net.eval() # lock voice model in exam mode so they don't learn

    
    dataset = MavCelebDataset(DATA_DIR) # load the heavy dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # load one person at a time

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR) # create folder if it doesn't exist

    print(f"Starting extraction for {len(dataset)} samples...")

    with torch.no_grad(): # turn off math tracking to save memory and speed it up
        for i, (face_img, voice_spec, label) in enumerate(loader):
            face_img, voice_spec = face_img.to(DEVICE), voice_spec.to(DEVICE) # move data to gpu

            #mbeddings (Vectors)
            face_emb = face_net(face_img) # squashes face image into 128 numbers
            voice_emb = voice_net(voice_spec.squeeze(1)) # squashes voice spectrogram into 128 numbers

            # data_ID.pt
            save_path = os.path.join(OUTPUT_DIR, f"sample_{i}.pt")
            torch.save({
                'face_emb': face_emb.cpu(), # pull face vector back to cpu to save
                'voice_emb': voice_emb.cpu(), # pull voice vector back to cpu
                'label': label.cpu() # pull the id label back
            }, save_path)

            if i % 100 == 0:
                print(f"Processed {i} samples...")

if __name__ == "__main__":
    extract()