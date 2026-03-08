import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os
from torch.utils.data import Dataset, DataLoader
from models import JointClassifier
from utils import orthogonal_projection_loss, EarlyStopping

# Config
EMBEDDING_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"
BATCH_SIZE = 32 
EPOCHS = 80 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# New Dataset Class that reads .pt files instead of images
class EmbeddingDataset(Dataset):
    def __init__(self, directory):
        self.files = glob.glob(os.path.join(directory, "*.pt"))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        # Squeeze to remove extra batch dim from saving
        return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label']

if __name__ == "__main__":
    # Load Data
    full_dataset = EmbeddingDataset(EMBEDDING_DIR)
    
    # Split Train/Test (Professor's note about keeping indices)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Set seed for reproducibility (Note: "sa le tin minte")
    generator = torch.Generator().manual_seed(42) 
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # We only train the Classifier now! (Encoders are frozen/done)
    # IMPORTANT: You need to know num_classes. 
    # For now, let's assume you pass it or calculate it.
    model = JointClassifier(num_classes=10).to(DEVICE) 
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for face_emb, voice_emb, labels in train_loader:
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE)
            
            # Simple Fusion
            fused_emb = face_emb + voice_emb
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)
            
            optimizer.zero_grad()
            logits = model(fused_emb)
            
            loss = criterion(logits, labels) + orthogonal_projection_loss(fused_emb, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase (simplified)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Check Early Stopping
        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
            
        # Save Checkpoint (Note: "sa salvez la 20 epochs")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "final_model.pth")