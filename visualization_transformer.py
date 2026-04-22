import umap
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.manifold import TSNE 
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset 
from models import ModalityTranslator, TransformerCrossAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"

def generate_transformer_visuals(mode='umap'):
    print(f"Loading models and data for Transformer {mode.upper()}...")

    full_dataset = EmbeddingDataset(TEST_DIR)
    
    # ensuring the exact 80/20 split used in training
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # all three components
    face_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    transformer_fusion = TransformerCrossAttention(embed_dim=512).to(DEVICE)

    
    checkpoint = torch.load("model_cu_transformer.pth", map_location=DEVICE)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    transformer_fusion.load_state_dict(checkpoint['transformer_fusion'])

    face_translator.eval()
    voice_translator.eval()
    transformer_fusion.eval()

    fused_vectors = []
    labels_list = []

    with torch.no_grad():
        for face_emb, voice_emb, labels in loader:
            face_emb, voice_emb = face_emb.to(DEVICE), voice_emb.to(DEVICE)

            # proiectam pe 512d
            f_512 = face_translator(face_emb)
            v_512 = voice_translator(voice_emb)

            # transf Attention to fuse them
            fused = transformer_fusion(f_512, v_512)
            
            #  first 30 IDs to keep graph clean
            mask = labels.view(-1) < 30
            if mask.sum() == 0: continue

            fused_vectors.append(fused[mask].cpu())
            labels_list.extend(labels[mask].numpy())

            if len(labels_list) > 1500: break

    all_fused = torch.cat(fused_vectors, dim=0).numpy()
    
    print(f'Running {mode.upper()} on fused embeddings...')
    if mode == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.5, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(all_fused)
    else:
        tsne = TSNE(n_components=2, perplexity=30, metric='cosine', init='pca', random_state=42)
        embeddings_2d = tsne.fit_transform(all_fused)

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels_list)
    cmap = plt.cm.get_cmap('nipy_spectral')

    for i, label in enumerate(unique_labels):
        idx = np.where(np.array(labels_list) == label)[0]
        color = cmap(i / len(unique_labels))
        #dot per sample, colored by identity
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=color, s=40, label=f'ID {label}', edgecolors='k', alpha=0.7)

    plt.title(f"{mode.upper()} - Fused Transformer Identities (Unseen Data)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"transformer_{mode}_identities.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_transformer_visuals(mode='umap')
    generate_transformer_visuals(mode='tsne')