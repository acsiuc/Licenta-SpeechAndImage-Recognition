import torch
import torch.nn.functional as F
import os
import glob
from models import ModalityTranslator, TransformerCrossAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_split(embeddings_dir, split_name):
    files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pt")))
    print(f"\nLoading {len(files)} pairs for {split_name}...")

    face_translator  = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)
    transformer_fusion = TransformerCrossAttention(embed_dim=512).to(DEVICE)

    checkpoint = torch.load("model_cu_transformer.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    transformer_fusion.load_state_dict(checkpoint['transformer_fusion'])

    face_translator.eval()
    voice_translator.eval()
    transformer_fusion.eval()

    face_embs  = []
    voice_embs = []

    with torch.no_grad():
        for f in files:
            data = torch.load(f, weights_only=False)
            face_embs.append(data['face_emb'].squeeze(0))
            voice_embs.append(data['voice_emb'].squeeze(0))

    face_embs  = torch.stack(face_embs).to(DEVICE)
    voice_embs = torch.stack(voice_embs).to(DEVICE)

    with torch.no_grad():
        f_512 = face_translator(face_embs)
        v_512 = voice_translator(voice_embs)

    # cosine similarity of true pairs (diagonal)
    true_sims = F.cosine_similarity(f_512, v_512)

    # full similarity matrix for retrieval
    sim_matrix = torch.matmul(f_512, v_512.T)

    # Rank-1: for each face, find the best matching voice
    best_matches = torch.argmax(sim_matrix, dim=1)
    correct = (best_matches == torch.arange(len(files)).to(DEVICE)).sum().item()
    rank1 = correct / len(files) * 100

    # Rank-5
    top5 = torch.topk(sim_matrix, k=5, dim=1).indices
    correct_r5 = sum(
        i in top5[i].tolist()
        for i in range(len(files))
    )
    rank5 = correct_r5 / len(files) * 100

    print(f"\n--- {split_name} RESULTS ---")
    print(f"Total pairs:              {len(files)}")
    print(f"Avg true-pair similarity: {true_sims.mean().item():.4f}")
    print(f"Rank-1 Retrieval:         {rank1:.2f}%")
    print(f"Rank-5 Retrieval:         {rank5:.2f}%")
    print(f"Random chance Rank-1:     {100/len(files):.2f}%")

if __name__ == "__main__":
    test_split(
        embeddings_dir = r"C:\Users\Axiuc\Downloads\mavceleb_test_english",
        split_name     = "MAVCeleb English Test"
    )
    test_split(
        embeddings_dir = r"C:\Users\Axiuc\Downloads\mavceleb_test_urdu",
        split_name     = "MAVCeleb Urdu Test (Cross-Lingual)"
    )