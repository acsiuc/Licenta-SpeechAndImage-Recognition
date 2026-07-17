import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset
from models import ModalityTranslator

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\youtube_embeddings"

def test_unseen_biometrics():
    print("Loading Unseen YouTube Data...")
    dataset = EmbeddingDataset(TEST_DIR)
    loader  = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    face_translator  = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)

    checkpoint = torch.load("model_arcface.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    face_translator.eval()
    voice_translator.eval()

    with torch.no_grad():
        for face_emb, voice_emb, labels in loader:
            face_emb = face_emb.to(DEVICE)
            voice_emb = voice_emb.to(DEVICE)
            labels = labels.to(DEVICE)

            f_512 = face_translator(face_emb)
            v_512 = voice_translator(voice_emb)

            # average face embeddings per identity
            unique_labels = labels.unique()
            n = len(unique_labels)

            avg_faces  = torch.zeros(n, 512).to(DEVICE)
            avg_voices = torch.zeros(n, 512).to(DEVICE)

            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                avg_faces[i]  = F.normalize(f_512[mask].mean(dim=0), p=2, dim=0)
                avg_voices[i] = F.normalize(v_512[mask].mean(dim=0), p=2, dim=0)

            # true match similarity
            true_sims = F.cosine_similarity(avg_faces, avg_voices)
            print(f"\nAverage Similarity of True Matches: {true_sims.mean().item():.4f}")

            # Rank-1 retrieval across identities
            sim_matrix   = torch.matmul(avg_faces, avg_voices.T)
            best_matches = torch.argmax(sim_matrix, dim=1)
            correct      = (best_matches == torch.arange(n).to(DEVICE)).sum().item()
            accuracy     = correct / n * 100

            print(f"\n--- TRUE IN-THE-WILD RESULTS (Rank-1 Retrieval) ---")
            print(f"Total Identities:          {n}")
            print(f"Correct Top-Choice Matches:{correct}")
            print(f"Zero-Shot Accuracy:        {accuracy:.2f}%")
            print(f"Random Chance:             {100/n:.2f}%")
            break

if __name__ == "__main__":
    test_unseen_biometrics()