import torch
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset 
from models import JointClassifier, ModalityTranslator
from utils import paeff_fusion

# 1. Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings" 
NUM_CLASSES = 1200 

def test_model():
    print("Loading test data and models...")
    
    full_dataset = EmbeddingDataset(TEST_DIR)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = JointClassifier(num_classes=NUM_CLASSES, embedding_dim=512).to(DEVICE)
    face_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)

    checkpoint = torch.load("final_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['classifier'])
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])

    model.eval()
    face_translator.eval()
    voice_translator.eval()

    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for face_emb, voice_emb, labels in test_loader:
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE)
            
            face_emb = face_translator(face_emb)
            voice_emb = voice_translator(voice_emb)
            
            fused_emb = paeff_fusion(face_emb, voice_emb)
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)
            
            logits = model(fused_emb)
            
            _, predicted_labels = torch.max(logits, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels.view(-1)).sum().item()

    accuracy = (correct_predictions / total_samples) * 100
    print(f" Total People Tested: {total_samples}")
    print(f" Correct Guesses: {correct_predictions}")
    print(f" Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test_model()