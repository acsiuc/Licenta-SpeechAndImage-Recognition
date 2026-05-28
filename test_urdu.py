import torch
import torch.nn.functional as F
import os
import glob
from models import ModalityTranslator

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"

def compute_eer(genuine_scores, impostor_scores):
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    thresholds = sorted(set(all_scores))
    best_eer = 1.0
    for t in thresholds:
        pred = [1 if s >= t else 0 for s in all_scores]
        fp = sum(1 for p, l in zip(pred, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(pred, all_labels) if p == 0 and l == 1)
        fpr = fp / max(1, len(impostor_scores))
        fnr = fn / max(1, len(genuine_scores))
        eer = (fpr + fnr) / 2
        if eer < best_eer:
            best_eer = eer
    return best_eer

def test():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    print(f"Loading {len(files)} Urdu test samples...")

    face_translator  = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)

    ckpt = torch.load("model_cu_transformer.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(ckpt['face_translator'])
    voice_translator.load_state_dict(ckpt['voice_translator'])
    face_translator.eval()
    voice_translator.eval()

    face_embs  = []
    voice_embs = []
    labels     = []

    with torch.no_grad():
        for f in files:
            data = torch.load(f, weights_only=False)
            face_embs.append(data['face_emb'].squeeze(0))
            voice_embs.append(data['voice_emb'].squeeze(0))
            labels.append(data['label'].item())

    face_embs  = torch.stack(face_embs).to(DEVICE)
    voice_embs = torch.stack(voice_embs).to(DEVICE)

    with torch.no_grad():
        f_512 = face_translator(face_embs)
        v_512 = voice_translator(voice_embs)

    n = len(files)
    true_sims = F.cosine_similarity(f_512, v_512)

    sim_matrix   = torch.matmul(f_512, v_512.T)
    best_matches = torch.argmax(sim_matrix, dim=1)
    correct      = (best_matches == torch.arange(n).to(DEVICE)).sum().item()
    rank1        = correct / n * 100

    # EER
    genuine_scores  = true_sims.cpu().tolist()
    impostor_scores = []
    sim_np = sim_matrix.cpu()
    for i in range(n):
        for j in range(n):
            if i != j:
                impostor_scores.append(sim_np[i][j].item())

    eer = compute_eer(genuine_scores, impostor_scores)

    print(f"\n--- URDU CROSS-LINGUAL RESULTS (Known Identities, Unseen Language) ---")
    print(f"Total identities:              {n}")
    print(f"Avg true-match similarity:     {true_sims.mean().item():.4f}")
    print(f"Rank-1 Retrieval:              {rank1:.2f}%")
    print(f"Random chance:                 {100/n:.2f}%")
    print(f"EER:                           {eer*100:.2f}%")
    print(f"MAVCeleb baseline EER (Urdu):  37.90%")

if __name__ == "__main__":
    test()