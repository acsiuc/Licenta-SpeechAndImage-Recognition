import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os
from torch.utils.data import Dataset, DataLoader
from models import JointClassifier, ModalityTranslator
from utils import orthogonal_projection_loss, EarlyStopping, cross_modal_alignment_loss, paeff_fusion

# Config
EMBEDDING_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"
BATCH_SIZE = 32 
EPOCHS = 80 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingDataset(Dataset):
    def __init__(self, directory):
        self.files = glob.glob(os.path.join(directory, "*.pt"))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
       
        return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label']

if __name__ == "__main__":
    
    full_dataset = EmbeddingDataset(EMBEDDING_DIR)
    

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42) 
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = JointClassifier(num_classes=1200).to(DEVICE) 
    face_translator = ModalityTranslator().to(DEVICE)
    voice_translator = ModalityTranslator().to(DEVICE)
    
    params_to_train = list(model.parameters()) + list(face_translator.parameters()) + list(voice_translator.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5)

    for epoch in range(EPOCHS):
        model.train()
        face_translator.train()  
        voice_translator.train() 
        total_loss = 0
        
        for face_emb, voice_emb, labels in train_loader:
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE)
            
            # translate the frozen vectors into aligned vectors 
            face_emb = face_translator(face_emb)
            voice_emb = voice_translator(voice_emb)
            
            #calculate Alignment Loss
            align_loss = cross_modal_alignment_loss(face_emb, voice_emb, labels)
            
            fused_emb = paeff_fusion(face_emb,voice_emb)
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)
            
            optimizer.zero_grad()
            logits = model(fused_emb)
            
        
            class_loss = criterion(logits, labels.view(-1))
            ortho_loss = orthogonal_projection_loss(fused_emb, labels)
            
            loss = class_loss + ortho_loss + (0.5 * align_loss)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        
        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
            
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "final_model.pth")

