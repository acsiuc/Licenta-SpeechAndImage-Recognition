import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, vgg16

EMBEDDING_DIM = 128

class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super(FaceEncoder, self).__init__()
        vgg= vgg16(pretrained = True)
        self.features = vgg.features #takes feature extration from vgg
        self.avgpool = vgg.avgpool #avg pooling layer
        
        self.projection = nn.Linear(512 * 7 * 7, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)# normalizing so we can dot product


class VoiceEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super(VoiceEncoder, self).__init__()
        self.resnet = resnet18(pretrained = False)#not pretrained like image, spectrograms look different
        self.resnet.conv1 = nn.Conv2d(1,64, kernel_size = 7, stride = 2, padding = 3, bias = False)#spectrogram expect 1 channel, replaces first layer to accept 1 input
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)# returns vector of 128 dimension

    def forward(self, x):
        x = self. resnet(x)
        return F.normalize(x, p=2, dim=1)

class JointClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim: int = 128):
        super(JointClassifier, self).__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

class ModalityTranslator(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super(ModalityTranslator, self).__init__()
        #this translates the vectors
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self,x):
        x = self.projector(x)
        return F.normalize(x, p=2, dim=1)