import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, vgg16

EMBEDDING_DIM = 128 # base size for our initial frozen vectors

class FaceEncoder(nn.Module):
    # the base model that looks at raw face images
    def __init__(self, embedding_dim: int = 128):
        super(FaceEncoder, self).__init__()
        vgg= vgg16(pretrained = True) # grab a vgg16 brain that already knows how to see shapes
        self.features = vgg.features # takes feature extraction from vgg
        self.avgpool = vgg.avgpool 
        
        self.projection = nn.Linear(512 * 7 * 7, embedding_dim) # squashes the giant vgg output down into our 128-number vector

    def forward(self, x):
        x = self.features(x) # pass image through vgg brain
        x = self.avgpool(x) 
        x = torch.flatten(x,1) # unroll the grid into a single flat line of numbers
        x = self.projection(x) # squash it through the 128d 
        return F.normalize(x, p=2, dim=1) # normalizing so the math doesn't explode when we dot product


class VoiceEncoder(nn.Module):
    # the base model that listens to raw voice spectrograms
    def __init__(self, embedding_dim: int = 128):
        super(VoiceEncoder, self).__init__()
        self.resnet = resnet18(pretrained = False) # not pretrained like image, spectrograms look completely different to ai
        self.resnet.conv1 = nn.Conv2d(1,64, kernel_size = 7, stride = 2, padding = 3, bias = False) # spectrograms only have 1 color channel, replace first layer to accept 1 input instead of 3
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim) # replaces the final layer to return our 128-dimensional vector

    def forward(self, x):
        x = self. resnet(x) # pass spectrogram through resnet brain
        return F.normalize(x, p=2, dim=1) # normalize the output vector

class JointClassifier(nn.Module):
    # the bouncer that guesses the final name
    def __init__(self, num_classes, embedding_dim: int = 512): # expanded 512d door to match our upgraded translators
        super(JointClassifier, self).__init__()
        # added a hidden layer to make the bouncer think deeper before guessing
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024), # inflate into a bigger thinking space
            nn.ReLU(), # activation function so it can learn complex patterns
            nn.Linear(1024, num_classes) # narrow it down to the exact number of buttons (identities)
        )

    def forward(self, x):
        return self.classifier(x) # make the final guess

class ModalityTranslator(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 512, dropout: float = 0.3):
        super(ModalityTranslator, self).__init__()
        
        # removed the old Tanh/Sigmoid attention here.
        # a deep, stable projector to get the 128d vectors up to 512d.
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.projector(x)
        return F.normalize(out, p=2, dim=1)

class TransformerCrossAttention(nn.Module):
    # mplement the Q, K, V math from "The Illustrated Transformer"
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.3):
        super(TransformerCrossAttention, self).__init__()
        
        # PyTorch's built-in Multihead Attention 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, face_emb, voice_emb):
        #  modalities into a sequence of length 2: [Face, Voice]
        # Shape becomes: (Batch_Size, 2_tokens, 512_dimensions)
        seq = torch.stack([face_emb, voice_emb], dim=1) 
        
        # Face and Voice act as Queries, Keys, and Values for each other
        attn_output, attn_weights = self.multihead_attn(query=seq, key=seq, value=seq)
        
        # add & normalize 
        seq_out = self.layer_norm(seq + self.dropout(attn_output))
        
        # average the two attention-boosted vectors together to get the final fused vector
        fused_emb = torch.mean(seq_out, dim=1)
        return F.normalize(fused_emb, p=2, dim=1)