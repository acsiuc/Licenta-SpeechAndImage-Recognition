import umap
import torch.nn.functional as F
import torch
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ModalityTranslator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"


def visualize_umap_urdu():
    print("Loading Urdu held-out test embeddings...")

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    print(f"Found {len(files)} Urdu test samples.")

    # ArcFace face encoder outputs 512D, not the legacy VGG16 128D.
    # This must match what model_arcface.pth's face_translator was
    # actually trained with, or load_state_dict will fail.
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
    all_vectors_tensor = all_vectors_tensor - all_vectors_tensor.mean(dim=0, keepdim=True)
    all_vectors_tensor = F.normalize(all_vectors_tensor, p=2, dim=1)
    all_vectors = all_vectors_tensor.numpy()

    print('Running UMAP...')
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.8,
        init='random',
        metric='cosine',
        random_state=42
    )

    embeddings_2d = reducer.fit_transform(all_vectors)

    n = len(labels_list)
    face_2d = embeddings_2d[:n]
    voice_2d = embeddings_2d[n:]
    plot_labels = labels_list

    print('Generating Plot...')
    plt.figure(figsize=(14, 10))

    unique_labels = np.unique(plot_labels)
    cmap = matplotlib.colormaps['nipy_spectral']

    for i, label in enumerate(unique_labels):
        idx = np.where(np.array(plot_labels) == label)[0]
        color = cmap(i / len(unique_labels))

        plt.scatter(face_2d[idx, 0], face_2d[idx, 1], color=color, marker='o', s=30,
                    label=f'ID{label} (Face)', alpha=0.85, edgecolors='k')
        plt.scatter(voice_2d[idx, 0], voice_2d[idx, 1], color=color, marker='^', s=30,
                    label=f'ID{label} (Urdu Voice)', alpha=0.85)

        f_point = face_2d[idx][0]
        v_point = voice_2d[idx][0]
        plt.plot([f_point[0], v_point[0]], [f_point[1], v_point[1]],
                 color=color, linestyle='--', alpha=0.3)

    plt.title("UMAP: ArcFace+ECAPA Fused Embeddings on Held-Out Urdu Test Set\n"
              "(64 known identities, unseen language, unseen recordings)")
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig("urdu_test_umap.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    visualize_umap_urdu()