import torch.nn.functional as F
import torch
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ModalityTranslator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"


def visualize_tsne_urdu():
    print("Loading Urdu held-out test embeddings...")

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    print(f"Found {len(files)} Urdu test samples.")

    # ArcFace face encoder outputs 512D, not the legacy VGG16 128D.
    # This must match the dimensions the translator was actually trained
    # with in model_arcface.pth, or load_state_dict will fail.
    face_translator = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)

    checkpoint = torch.load("model_arcface.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])

    face_translator.eval()
    voice_translator.eval()

    face_vectors = []
    voice_vectors = []
    labels_list = []

    print('Extracting and aligning embeddings...')
    with torch.no_grad():
        for f in files:
            data = torch.load(f, weights_only=False)
            face_emb = data['face_emb'].to(DEVICE)
            voice_emb = data['voice_emb'].to(DEVICE)
            label = data['label'].item()

            f_512 = face_translator(face_emb).cpu()
            v_512 = voice_translator(voice_emb).cpu()

            f_512 = F.normalize(f_512, p=2, dim=1)
            v_512 = F.normalize(v_512, p=2, dim=1)

            face_vectors.append(f_512)
            voice_vectors.append(v_512)
            labels_list.append(label)

    all_f = torch.cat(face_vectors, dim=0)
    all_v = torch.cat(voice_vectors, dim=0)

    all_vectors_tensor = torch.cat([all_f, all_v], dim=0)
    # center so the face and voice clouds overlap rather than forming
    # two separate distant clusters purely from modality offset
    all_vectors_tensor = all_vectors_tensor - all_vectors_tensor.mean(dim=0, keepdim=True)
    all_vectors_tensor = F.normalize(all_vectors_tensor, p=2, dim=1)
    all_vectors = all_vectors_tensor.numpy()

    print('Running t-SNE...')
    tsne = TSNE(
        n_components=2,
        perplexity=25,
        early_exaggeration=24.0,
        metric='cosine',
        init='pca',
        learning_rate='auto',
        max_iter=2500,
        random_state=42
    )

    embeddings_2d = tsne.fit_transform(all_vectors)

    n = len(labels_list)
    face_2d = embeddings_2d[:n]
    voice_2d = embeddings_2d[n:]
    plot_labels = np.array(labels_list)

    plt.figure(figsize=(12, 9))

    unique_labels = np.unique(plot_labels)

    # 64 fully-distinct colors is unreadable. Instead: plot every identity
    # as a neutral gray background point (so all data is still shown),
    # then highlight a small, fixed subset in strong distinct colors so
    # the eye can actually follow individual genuine pairs.
    n_highlight = 10
    rng = np.random.RandomState(24)
    highlight_labels = rng.choice(unique_labels, size=min(n_highlight, len(unique_labels)),
                                   replace=False)
    highlight_cmap = matplotlib.colormaps['tab10']

    # background: all 64 identities, low-emphasis gray
    plt.scatter(face_2d[:, 0], face_2d[:, 1], color='lightgray', marker='o', s=25,
                alpha=0.5, edgecolors='none', zorder=1)
    plt.scatter(voice_2d[:, 0], voice_2d[:, 1], color='lightgray', marker='^', s=25,
                alpha=0.5, edgecolors='none', zorder=1)

    # foreground: highlighted subset, distinct colors, connector lines
    for i, label in enumerate(highlight_labels):
        idx = np.where(plot_labels == label)[0]
        color = highlight_cmap(i)

        plt.scatter(face_2d[idx, 0], face_2d[idx, 1], color=color, marker='o', s=90,
                    label=f'ID{label} (Face)', alpha=0.95, edgecolors='k', linewidths=0.8,
                    zorder=3)
        plt.scatter(voice_2d[idx, 0], voice_2d[idx, 1], color=color, marker='^', s=90,
                    label=f'ID{label} (Urdu Voice)', alpha=0.95, edgecolors='k', linewidths=0.8,
                    zorder=3)

        f_point = face_2d[idx][0]
        v_point = voice_2d[idx][0]
        plt.plot([f_point[0], v_point[0]], [f_point[1], v_point[1]],
                 color=color, linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)

    plt.title("t-SNE: ArcFace+ECAPA Fused Embeddings on Held-Out Urdu Test Set\n"
              "(64 known identities, unseen language, unseen recordings — "
              f"{len(highlight_labels)} identities highlighted for clarity)")
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    plt.savefig("urdu_test_tsne.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    visualize_tsne_urdu()