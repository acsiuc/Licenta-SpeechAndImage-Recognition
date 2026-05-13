import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset 
from models import ModalityTranslator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
TEST_DIR = r"C:\Users\Axiuc\Downloads\youtube_embeddings" 

def test_unseen_biometrics():
    print("Loading Unseen YouTube Data...")
    dataset = EmbeddingDataset(TEST_DIR) 
    # Load all 80 samples at once so we can do a massive cross-comparison lineup
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    face_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE) 
    voice_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE) 

    print("Loading Biometric Translators...")
    checkpoint = torch.load("model_cu_transformer.pth", map_location=DEVICE) 
    face_translator.load_state_dict(checkpoint['face_translator']) 
    voice_translator.load_state_dict(checkpoint['voice_translator']) 

    face_translator.eval() 
    voice_translator.eval()
    
    with torch.no_grad(): 
        for face_emb, voice_emb, labels in loader:
            face_emb = face_emb.to(DEVICE)
            voice_emb = voice_emb.to(DEVICE)
            labels = labels.to(DEVICE)

            # Translate to 512D space
            f_512 = face_translator(face_emb) 
            v_512 = voice_translator(voice_emb) 
            
            # 1. Let's look at the raw numbers!
            true_match_sims = F.cosine_similarity(f_512, v_512)
            print(f"\nAverage Similarity of True Matches: {true_match_sims.mean().item():.4f}")
            
            # 2. The Lineup Test (Rank-1 Retrieval)
            # Compare every face against EVERY voice in the room
            sim_matrix = torch.matmul(f_512, v_512.T)
            
            # For each face, find the index of the voice that scored the absolute highest
            best_match_indices = torch.argmax(sim_matrix, dim=1)
            
            # Grab the ID labels of those winning voices
            predicted_labels = labels[best_match_indices]
            
            # Check if the winning voice's ID matches the face's ID
            correct_predictions = (predicted_labels == labels).sum().item()
            total_tests = labels.size(0)

            accuracy = (correct_predictions / total_tests) * 100 
            
            print(f"\n--- TRUE IN-THE-WILD RESULTS (Rank-1 Retrieval) ---")
            print(f"Total Lineups Conducted: {total_tests}")
            print(f"Correct Top-Choice Matches: {correct_predictions}")
            print(f"Zero-Shot Accuracy: {accuracy:.2f}%")
            
            break # We only need one batch since it loaded all 80

if __name__ == "__main__":
    test_unseen_biometrics()