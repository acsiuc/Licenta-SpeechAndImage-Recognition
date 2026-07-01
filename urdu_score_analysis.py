import torch
import torch.nn.functional as F
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
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

    genuine_scores = []
    impostor_scores = []
    for i in range(n):
        for j in range(n):
            if i == j:
                genuine_scores.append(sim_matrix[i, j])
            else:
                impostor_scores.append(sim_matrix[i, j])

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    return genuine_scores, impostor_scores


def plot_histogram(genuine_scores, impostor_scores):
    plt.figure(figsize=(10, 6))

    plt.hist(genuine_scores, bins=20, alpha=0.6, color='green',
              label=f'Genuine pairs (n={len(genuine_scores)})', density=True)
    plt.hist(impostor_scores, bins=20, alpha=0.6, color='red',
              label=f'Impostor pairs (n={len(impostor_scores)})', density=True)

    plt.axvline(genuine_scores.mean(), color='darkgreen', linestyle='--', linewidth=1.5,
                label=f'Genuine mean = {genuine_scores.mean():.4f}')
    plt.axvline(impostor_scores.mean(), color='darkred', linestyle='--', linewidth=1.5,
                label=f'Impostor mean = {impostor_scores.mean():.4f}')

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Genuine vs. Impostor Score Distribution\n'
              'ArcFace+ECAPA Fused Embeddings, Held-Out Urdu Test Set (64 identities)')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig('urdu_score_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_roc(genuine_scores, impostor_scores):
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Chance (AUC = 0.5)')

    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('ROC Curve\nArcFace+ECAPA Fused Embeddings, Held-Out Urdu Test Set')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig('urdu_roc_curve.png', bbox_inches='tight', dpi=300)
    plt.show()

    return auc


if __name__ == "__main__":
    genuine_scores, impostor_scores = evaluate_scores()

    print(f"\nGenuine pairs: {len(genuine_scores)}, mean = {genuine_scores.mean():.4f}")
    print(f"Impostor pairs: {len(impostor_scores)}, mean = {impostor_scores.mean():.4f}")
    print(f"Separation gap: {genuine_scores.mean() - impostor_scores.mean():.4f}")

    plot_histogram(genuine_scores, impostor_scores)
    auc = plot_roc(genuine_scores, impostor_scores)

    print(f"\nROC-AUC: {auc:.4f}")