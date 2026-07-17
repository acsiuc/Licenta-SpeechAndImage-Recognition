import torch
import torch.nn.functional as F
import os
import glob
import sys
sys.path.append(r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta")
# this is why "from models import ModalityTranslator" works even though this script
# lives in tests_after_training/, a subfolder — models.py isnt in the same directory,
# so the path has to be added manually. hardcoded to your machine, this is the FIRST thing
# to fix if running from anywhere else, script wont even import without it
from models import ModalityTranslator

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"
# second hardcoded path, needs to exist and be populated for this to run at all

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
        fnr = fn / max(1, len(genuine_scores))   # fixed
        eer = (fpr + fnr) / 2
        if eer < best_eer:
            best_eer = eer
    return best_eer
    # NOTE: this isnt the textbook EER definition (point where FPR == FNR exactly)
    # its "minimum average of FPR and FNR across observed thresholds" — close enough in practice
    # and a defensible approximation, but if asked "is this the exact EER crossing point",
    # the honest answer is no, its the best operating point found by sweeping observed scores as thresholds

def test():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    print(f"Loading {len(files)} Urdu test samples...")

    face_translator  = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)

    ckpt = torch.load("model_arcface.pth", map_location=DEVICE, weights_only=False)
    # relative path — this loads from whatever the CURRENT WORKING DIRECTORY is when you run the script,
    # not from tests_after_training/ or the models.py folder. confirm the checkpoint is sitting
    # wherever you'll actually be cd'd into when you run this live
    # also: this loads "model_arcface.pth" but train.py saves "model_cu_transformer.pth" — name mismatch,
    # resolve which file actually exists before the live run
    face_translator.load_state_dict(ckpt['face_translator'])
    voice_translator.load_state_dict(ckpt['voice_translator'])
    face_translator.eval()
    voice_translator.eval()
    # only translators are loaded from the checkpoint — no classifier, no transformer_fusion
    # this is the concrete proof that fusion is training-only, the eval path never touches it
    # even though the checkpoint dict (per train.py) DOES contain fusion weights, they just go unused here

    face_embs  = []
    voice_embs = []
    labels     = []

    with torch.no_grad():
        for f in files:
            try:
                data = torch.load(f, weights_only=False)
            except Exception as e:
                print(f"FAILED on file: {f}")
                print(f"Error: {e}")
                continue
            face_embs.append(data['face_emb'].squeeze(0))
            voice_embs.append(data['voice_emb'].squeeze(0))
            labels.append(data['label'].item())
    # unlike extract_features_all.py, failures here ARE printed with the filename and error,
    # much better visibility than the silent except in the extraction script

    face_embs  = torch.stack(face_embs).to(DEVICE)
    voice_embs = torch.stack(voice_embs).to(DEVICE)

    with torch.no_grad():
        f_512 = face_translator(face_embs)
        v_512 = voice_translator(voice_embs)

    n = len(face_embs)
    # n is derived from the actually-loaded embeddings, not from the original files list —
    # correct, accounts for any files that failed to load above
    true_sims = F.cosine_similarity(f_512, v_512)
    # this is the genuine-pair similarity: face[i] vs voice[i], same identity by construction of the file ordering

    sim_matrix   = torch.matmul(f_512, v_512.T)
    # since both f_512 and v_512 are L2-normalized (ModalityTranslator.forward normalizes on the way out),
    # this matmul IS cosine similarity, not raw dot product — same math as F.cosine_similarity above just batched
    best_matches = torch.argmax(sim_matrix, dim=1)
    correct      = (best_matches == torch.arange(n).to(DEVICE)).sum().item()
    rank1        = correct / n * 100
    # Rank-1: for each face, is its highest-similarity voice the correct one out of all n candidates
    # this is the 79.69% number, n=64 known identities here

    # EER
    genuine_scores  = true_sims.cpu().tolist()
    impostor_scores = []
    sim_np = sim_matrix.cpu()
    for i in range(n):
        for j in range(n):
            if i != j:
                impostor_scores.append(sim_np[i][j].item())
    # every off-diagonal entry becomes an impostor score, so n^2 - n impostor pairs vs n genuine pairs
    # at n=64 thats 4032 impostor scores against 64 genuine — heavily imbalanced by design,
    # which is normal for verification-style EER, not a bug

    eer = compute_eer(genuine_scores, impostor_scores)

    print(f"\n--- URDU CROSS-LINGUAL RESULTS (Known Identities, Unseen Language) ---")
    print(f"Total identities:              {n}")
    print(f"Avg true-match similarity:     {true_sims.mean().item():.4f}")
    print(f"Rank-1 Retrieval:              {rank1:.2f}%")
    print(f"Random chance:                 {100/n:.2f}%")
    print(f"EER:                           {eer*100:.2f}%")
    print(f"MAVCeleb baseline EER (Urdu):  37.90%")
    # this last line is hardcoded, not computed — its a fixed reference number from the MAVCeleb paper/baseline
    # good to know its a literal constant here, not pulled from anywhere live, if asked "where does 37.90% come from"

if __name__ == "__main__":
    test()