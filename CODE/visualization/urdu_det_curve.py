import torch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import det_curve
from models import ModalityTranslator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"


def evaluate_scores():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    print(f"Loading {len(files)} Urdu test samples...")

    face_translator = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)

    checkpoint = torch.load("model_arcface.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])

    face_translator.eval()
    voice_translator.eval()

    face_embs, voice_embs = [], []
    with torch.no_grad():
        for f in files:
            data = torch.load(f, weights_only=False)
            face_embs.append(data['face_emb'].squeeze(0))
            voice_embs.append(data['voice_emb'].squeeze(0))

    face_embs = torch.stack(face_embs).to(DEVICE)
    voice_embs = torch.stack(voice_embs).to(DEVICE)

    with torch.no_grad():
        f_512 = face_translator(face_embs)
        v_512 = voice_translator(voice_embs)

    # both translators already L2-normalize their output (see models.py),
    # so this dot product is mathematically identical to cosine similarity
    sim_matrix = torch.matmul(f_512, v_512.T).cpu().numpy()
    n = sim_matrix.shape[0]

    scores = []
    labels = []
    for i in range(n):
        for j in range(n):
            scores.append(sim_matrix[i, j])
            labels.append(1 if i == j else 0)

    return np.array(scores), np.array(labels)


def plot_det(scores, labels):
    # det_curve returns FPR and FNR directly on a linear [0,1] scale;
    # the probit rescaling below is what turns this into the standard
    # DET plot (normal-deviate scale), matching the convention used in
    # speaker verification literature (e.g. Desplanques et al., ECAPA-TDNN)
    fpr, fnr, thresholds = det_curve(labels, scores)

    from scipy.stats import norm

    # clip to avoid -inf/+inf at exactly 0 or 1
    eps = 1e-4
    fpr_clipped = np.clip(fpr, eps, 1 - eps)
    fnr_clipped = np.clip(fnr, eps, 1 - eps)

    x = norm.ppf(fpr_clipped)
    y = norm.ppf(fnr_clipped)

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, color='steelblue', linewidth=2, label='DET curve')

    # tick labels in real probability terms, positioned via probit transform
    tick_probs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    tick_positions = norm.ppf(tick_probs)
    tick_labels = [f'{p * 100:.0f}%' for p in tick_probs]

    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    plt.xlim(norm.ppf(0.005), norm.ppf(0.95))
    plt.ylim(norm.ppf(0.005), norm.ppf(0.95))

    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve\nArcFace+ECAPA Fused Embeddings, Held-Out Urdu Test Set')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig('urdu_det_curve.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    scores, labels = evaluate_scores()
    plot_det(scores, labels)