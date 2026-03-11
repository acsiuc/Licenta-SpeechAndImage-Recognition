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
        self.avgpool = vgg.avgpool # avg pooling layer to shrink the grid
        
        self.projection = nn.Linear(512 * 7 * 7, embedding_dim) # squashes the giant vgg output down into our 128-number vector

    def forward(self, x):
        x = self.features(x) # pass image through vgg brain
        x = self.avgpool(x) # pool it down
        x = torch.flatten(x,1) # unroll the grid into a single flat line of numbers
        x = self.projection(x) # squash it through the 128d doorway
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
    # the detective that aligns and upgrades our features
    def __init__(self, input_dim: int = 128, output_dim: int = 512):
        super(ModalityTranslator, self).__init__()
        # takes the frozen 128d input and inflates it into a massive 512d space
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(), # non-linear math to help it untangle tricky identities
            nn.Linear(512, output_dim) 
        )
    def forward(self,x):
        x = self.projector(x) # push the vector through the upgrade path
        return F.normalize(x, p=2, dim=1) # normalize the new 512d vector so it plays nice with others