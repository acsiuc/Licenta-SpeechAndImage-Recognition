import umap
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.manifold import TSNE 
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset 
from models import ModalityTranslator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR  =r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"

def visualize_umap():
    print("Loading Data...")

    dataset = EmbeddingDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size = 256, shuffle = True)

    face_translator = ModalityTranslator(input_dim = 128, output_dim = 512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim = 128, output_dim = 512).to(DEVICE)

    # using the fixed model where the translators actually learned something
    checkpoint = torch.load("final_model_modality_translators_fixed.pth", map_location=DEVICE)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])

    face_translator.eval()
    voice_translator.eval()

    face_vectors = []
    voice_vectors = []
    labels_list = []

    print('Extracting and aligning embeddings...')
    with torch.no_grad():
        for face_emb, voice_emb, labels in loader:
            face_emb, voice_emb = face_emb.to(DEVICE), voice_emb.to(DEVICE)

            f_512 = face_translator(face_emb).cpu()
            v_512 = voice_translator(voice_emb).cpu()

            # normalize so they live on the same scale
            f_512 = F.normalize(f_512, p=2, dim=1).cpu()
            v_512 = F.normalize(v_512, p=2, dim=1).cpu()

            mask = labels.view(-1) < 50
            if mask.sum() == 0: continue

            face_vectors.append(f_512[mask])
            voice_vectors.append(v_512[mask])
            labels_list.extend(labels[mask].numpy())

            if len(labels_list) > 2000: break

    all_f = torch.cat(face_vectors, dim=0)
    all_v = torch.cat(voice_vectors, dim=0)

    all_vectors_tensor = torch.cat([all_f, all_v], dim=0)

    all_vectors_tensor = all_vectors_tensor - all_vectors_tensor.mean(dim = 0, keepdim= True)
    all_vectors_tensor = F.normalize(all_vectors_tensor, p=2, dim=1)
    all_vectors = all_vectors_tensor.numpy()
    
    print('Running UMAP...')

    reducer = umap.UMAP(
        n_neighbors=30, 
        min_dist=0.3, 
        metric='cosine', 
        random_state=42
    )
    
    embeddings_2d = reducer.fit_transform(all_vectors)

    half = len(embeddings_2d) // 2
    face_2d = embeddings_2d[:half]
    voice_2d = embeddings_2d[half:]
    plot_labels = labels_list

    print('Generating Plot...')
    plt.figure(figsize = (14,10))

    unique_labels = np.unique(plot_labels)
    cmap = plt.cm.get_cmap('nipy_spectral') 

    for i, label in enumerate(unique_labels):
        idx = np.where(np.array(plot_labels) == label)[0]
        color = cmap(i / len(unique_labels))

        plt.scatter(face_2d[idx,0], face_2d[idx, 1], color=color, marker ='o',s=15, label = f'ID{label} (Face)', alpha = 0.8, edgecolors ='k')
        plt.scatter(voice_2d[idx,0], voice_2d[idx,1], color=color, marker = '^',s=15, label = f'ID{label} (Voice)', alpha = 0.8)

        f_mean = face_2d[idx].mean(axis=0)
        v_mean = voice_2d[idx].mean(axis=0)
        plt.plot([f_mean[0], v_mean[0]], [f_mean[1], v_mean[1]], color=color, linestyle ='--', alpha = 0.3)

    plt.title("UMAP Visualization of Aligned Modalities") 
    # plt.legend(loc = 'upper right', bbox_to_anchor = (1.20, 1), ncol=1)
    plt.grid(True, linestyle=':', alpha = 0.6)

    plt.savefig("face_voice_umap.png", bbox_inches = 'tight', dpi=300) 
    plt.show()

if __name__ == "__main__":
    visualize_umap()