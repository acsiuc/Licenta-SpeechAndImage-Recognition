import os
import glob
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class MavCelebDataset(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        
        self.faceRoot = os.path.join(rootDir, "faces") 
        self.voiceRoot = os.path.join(rootDir, "voices") 

        if not os.path.exists(self.faceRoot) or not os.path.exists(self.voiceRoot):
            raise RuntimeError(f"Error: Could not find 'faces' or 'voices' inside {rootDir}")
 
        self.identities = sorted([d for d in os.listdir(self.faceRoot) if d.startswith('id')])

        self.dataMap = {}
        print(f"Scanning {len(self.identities)} identity folders...")
        
        for identity in self.identities:
            facePaths = glob.glob(os.path.join(self.faceRoot, identity, "**", "*.jpg"), recursive=True)
            voicePaths = glob.glob(os.path.join(self.voiceRoot, identity, "**", "*.wav"), recursive=True)
            
            if len(facePaths) > 0 and len(voicePaths) > 0:
                self.dataMap[identity] = {"audios": voicePaths, "faces": facePaths}

        self.validIds = list(self.dataMap.keys())
        self.classToIdx = {identity: i for i, identity in enumerate(self.validIds)}
        
        print(f"Found {len(self.validIds)} valid identities with both audio and video")

        # setup spectrogram tool — log-scale (dB) is standard: raw mel power has huge dynamic range
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        # setup face transform tool
        self.face_transform = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomRotation(15), # NEW: Randomly tilt the head up to 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # NEW: Randomly change lighting and shadows
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def __len__(self):
        return len(self.validIds)*500  # artificially extending dataset

    def __getitem__(self, idx):
        anchorId = self.validIds[idx % len(self.validIds)]
        label = self.classToIdx[anchorId] # get their number tag

        # picks the file 
        facePath = random.choice(self.dataMap[anchorId]["faces"])
        voicePath = random.choice(self.dataMap[anchorId]["audios"])

        faceImg = Image.open(facePath).convert("RGB") # opens and converts
        faceTensor = self.face_transform(faceImg) # transforms face into ready-to-use numbers

        waveform, sr = torchaudio.load(voicePath) # load audio

        if waveform.shape[0] > 1: # If it has 2 channels (Stereo)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000: # resample if needed
            resampler = torchaudio.transforms.Resample(sr, 16000) # standardizes the audio quality
            waveform = resampler(waveform)

        targetLen = 16000 * 3 # aim for exactly 3 seconds of audio
        
        if waveform.shape[1] < targetLen: # padding if too short
            waveform = F.pad(waveform, (0, targetLen - waveform.shape[1])) # add silence to the end if it's too short
        else:
            #grab a random 3-second window instead of the first 3 seconds
            max_start = waveform.shape[1] - targetLen
            start_idx = random.randint(0, max_start)
            waveform = waveform[:, start_idx: start_idx + targetLen] # cropping if too long # cut it off at 3 seconds if it's too long

        specTensor = self.mel_transform(waveform)       # create mel spectrogram (linear power scale)
        specTensor = self.amplitude_to_db(specTensor)  # convert to dB (log scale) — tighter dynamic range, better gradients
        specTensor = specTensor.unsqueeze(0)            # add channel dim: [1, 1, n_mels, time]

        return faceTensor, waveform.squeeze(0), specTensor, torch.tensor(label, dtype=torch.long)

class EmbeddingDataset(Dataset):
    def __init__(self, directory):
        self.files = glob.glob(os.path.join(directory, "*.pt"))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label']