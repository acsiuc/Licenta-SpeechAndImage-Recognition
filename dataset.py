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
        
        # setting up paths
        self.faceRoot = os.path.join(rootDir, "faces", "English")
        self.voiceRoot = os.path.join(rootDir, "voices", "English")

        # check if main folders exist
        if not os.path.exists(self.faceRoot) or not os.path.exists(self.voiceRoot):
            raise RuntimeError(f"Error: Could not find 'faces/English' or 'voices/English' inside {rootDir}")

        # scanning for jpg files directly
        faceFiles = glob.glob(os.path.join(self.faceRoot, "*.jpg"))
        
        # to get just the ID 
        self.identities = sorted([os.path.splitext(os.path.basename(f))[0] for f in faceFiles])

        self.classToIdx = {clsName: i for i, clsName in enumerate(self.identities)}

        self.dataMap = {}
        
        print(f"Scanning {len(self.identities)} identities...")
        
        for identity in self.identities:
            # constructing paths for face and voice directly
            facePath = os.path.join(self.faceRoot, identity + ".jpg")
            voicePath = os.path.join(self.voiceRoot, identity + ".wav")

            # checking if both exist before adding
            if os.path.exists(facePath) and os.path.exists(voicePath):
                # storing as lists 
                self.dataMap[identity] = {"audios": [voicePath], "faces": [facePath]}

        self.validIds = list(self.dataMap.keys())
        print(f"Found {len(self.validIds)} valid identities with both audio and video")

        # setup spectrogram tool
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
        
        # setup face transform tool
        self.face_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.validIds)  #*50 artificially extending dataset

    def __getitem__(self, idx):
        # pick random person
        anchorId = random.choice(self.validIds)
        label = self.classToIdx[anchorId]

        # picks the file 
        facePath = random.choice(self.dataMap[anchorId]["faces"])
        voicePath = random.choice(self.dataMap[anchorId]["audios"])

        faceImg = Image.open(facePath).convert("RGB") # opens and converts
        faceTensor = self.face_transform(faceImg) # transforms face

        waveform, sr = torchaudio.load(voicePath) # load audio
        if sr != 16000: # resample if needed
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        targetLen = 16000 * 3
        
        if waveform.shape[1] < targetLen: # padding if too short
            waveform = F.pad(waveform, (0, targetLen - waveform.shape[1]))
        else: 
            waveform = waveform[:, :targetLen] # cropping if too long

        specTensor = self.mel_transform(waveform) # create spectrogram
        specTensor = specTensor.unsqueeze(0) # add channel dim

        return faceTensor, specTensor, torch.tensor(label, dtype=torch.long)

class EmbeddingDataset(Dataset):

    def __init__(self, directory):
        # Find all .pt files
        self.files = glob.glob(os.path.join(directory, "*.pt"))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the dictionary we saved earlier
        data = torch.load(self.files[idx])
        
        # Return the vectors. 
        # .squeeze(0) removes the extra dimension we added during extraction
        return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label']