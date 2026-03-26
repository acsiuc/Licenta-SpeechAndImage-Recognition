import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os
from torch.utils.data import Dataset, DataLoader
from models import JointClassifier, ModalityTranslator
from utils import orthogonal_projection_loss, EarlyStopping, cross_modal_alignment_loss, paeff_fusion

EMBEDDING_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings" # path to our pre-baked 128d vectors
BATCH_SIZE = 32 
EPOCHS = 80 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available

class EmbeddingDataset(Dataset):
    def __init__(self, directory):
        self.files = glob.glob(os.path.join(directory, "*.pt")) # grab all the tiny .pt files
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx]) # open the specific file
       
        return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label'] # return the frozen vectors and the name tag

if __name__ == "__main__":
    
    full_dataset = EmbeddingDataset(EMBEDDING_DIR)
    

    train_size = int(0.8 * len(full_dataset)) # 80 percent for training pile
    test_size = len(full_dataset) - train_size # 20 percent for testing pile
    
    generator = torch.Generator().manual_seed(42)  # locks the random shuffle so we always get the exact same split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = JointClassifier(num_classes=1200,embedding_dim=512).to(DEVICE) # spawn the main classifier with the upgraded 512d brain
    face_translator = ModalityTranslator(input_dim = 128, output_dim = 512).to(DEVICE) # face detective upgrading 128 to 512
    voice_translator = ModalityTranslator(input_dim = 128, output_dim = 512).to(DEVICE) # voice detective upgrading 128 to 512
    
    params_to_train = list(model.parameters()) + list(face_translator.parameters()) + list(voice_translator.parameters()) # group all the brains together
    optimizer = optim.Adam(model.parameters(), lr=0.001) # sets up the learning rate
    criterion = nn.CrossEntropyLoss() # the math that calculates if we guessed the name right
    early_stopper = EarlyStopping(patience=15) # giving the bigger brain more patience to settle down

    for epoch in range(EPOCHS):
        model.train() # turn on learning mode for classifier
        face_translator.train()  # turn on learning mode for face detective
        voice_translator.train() # turn on learning mode for voice detective
        total_loss = 0
        
        for face_emb, voice_emb, labels in train_loader:
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE) # push to gpu
            
            # translate the frozen vectors into aligned vectors 
            face_emb = face_translator(face_emb) # push face through the 512d doorway
            voice_emb = voice_translator(voice_emb) # push voice through the 512d doorway
            
            #calculate Alignment Loss
            align_loss = cross_modal_alignment_loss(face_emb, voice_emb, labels) # penalty if face and voice point in diff directions
            
            fused_emb = paeff_fusion(face_emb,voice_emb) # smart mix the two vectors together
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1) # normalize so the math doesn't explode
            
            optimizer.zero_grad() # wipe the memory from the last batch
            logits = model(fused_emb) # bouncer makes a guess
            
        
            class_loss = criterion(logits, labels.view(-1)) # penalty for guessing the wrong name
            ortho_loss = orthogonal_projection_loss(fused_emb, labels) # penalty if different people are too close together in math space

            if epoch == 0 and total_loss == 0:  # print a snapshot of the very first batch
                print(f"Class Loss: {class_loss.item():.4f}")
                print(f"Ortho Loss: {ortho_loss.item():.4f}")
                print(f"Align Loss: {align_loss.item():.4f}")
              
                
            
            loss = class_loss + (0.1*ortho_loss) + (0.1 * align_loss) # master grading rubric - turned down side quests to 0.1 so it focuses on names
            
            loss.backward() # send the penalties backwards through the network
            optimizer.step() # actually update the weights/brains
            total_loss += loss.item()

        
        avg_loss = total_loss / len(train_loader) # calculate the average score for the whole epoch
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        
        early_stopper(avg_loss) # check if we stopped improving
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
            
        if (epoch + 1) % 20 == 0: # save a backup every 20 epochs just in case
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    checkpoint = {
        'classifier': model.state_dict(), # pack classifier brain
        'face_translator': face_translator.state_dict(), # pack face translator brain
        'voice_translator': voice_translator.state_dict() # pack voice translator brain
    }
    torch.save(checkpoint, "final_model.pth") # save the fully packed suitcase for test.py