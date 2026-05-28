import torch
import torch.nn.functional as F
import os
import glob
from models import ModalityTranslator, TransformerCrossAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENGLISH_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_test_english"
URDU_DIR    = r"C:\Users\Axiuc\Downloads\mavceleb_test_urdu"

def test_generalization(embeddings_dir, split_name):
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

    # genuine pair similarities (diagonal — same index = same person)
    genuine_sims = F.cosine_similarity(f_512, v_512)

    # impostor similarities — compare each face against all other voices
    sim_matrix = torch.matmul(f_512, v_512.T)
    
    # mask out diagonal (genuine pairs)
    mask = ~torch.eye(len(files), dtype=torch.bool).to(DEVICE)
    impostor_sims = sim_matrix[mask]

    print(f"\n--- {split_name} GENERALIZATION RESULTS ---")
    print(f"Total pairs:                    {len(files)}")
    print(f"Genuine pair avg similarity:    {genuine_sims.mean().item():.4f}")
    print(f"Impostor pair avg similarity:   {impostor_sims.mean().item():.4f}")
    print(f"Separation gap:                 {(genuine_sims.mean() - impostor_sims.mean()).item():.4f}")
    print(f"Genuine min:                    {genuine_sims.min().item():.4f}")
    print(f"Genuine max:                    {genuine_sims.max().item():.4f}")
    print(f"Impostor max:                   {impostor_sims.max().item():.4f}")

if __name__ == "__main__":
    test_generalization(ENGLISH_DIR, "MAVCeleb English (Closed-Set Generalization)")
    test_generalization(URDU_DIR,    "MAVCeleb Urdu (Cross-Lingual Generalization)")