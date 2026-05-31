import torch
import torch.nn.functional as F
import os
import glob
import random
from models import ModalityTranslator, TransformerCrossAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"

def compute_eer(genuine_scores, impostor_scores):
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    thresholds = sorted(set(all_scores))
    best_eer = 1.0
    
    for thresh in thresholds:
        predicted = [1 if s >= thresh else 0 for s in all_scores]
        fp = sum(1 for p, l in zip(predicted, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predicted, all_labels) if p == 0 and l == 1)
        fpr = fp / max(1, len(impostor_scores))
        fnr = fn / max(1, len(genuine_scores))
        eer = (fpr + fnr) / 2
        if eer < best_eer:
            best_eer = eer
    return best_eer

def test_verification():
    files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "*.pt")))
    print(f"Loading {len(files)} embeddings...")

    face_translator   = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator  = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)
    transformer_fusion = TransformerCrossAttention(embed_dim=512).to(DEVICE)

    checkpoint = torch.load("model_cu_transformer.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    transformer_fusion.load_state_dict(checkpoint['transformer_fusion'])

    face_translator.eval()
    voice_translator.eval()
    transformer_fusion.eval()

    # group embeddings by identity label
    identity_map = {}
    with torch.no_grad():
        for f in files:
            data  = torch.load(f, weights_only=False)
            label = int(data['label'].item())
            face_emb  = face_translator(data['face_emb'].to(DEVICE)).cpu()
            voice_emb = voice_translator(data['voice_emb'].to(DEVICE)).cpu()
            if label not in identity_map:
                identity_map[label] = {'faces': [], 'voices': []}
            identity_map[label]['faces'].append(face_emb)
            identity_map[label]['voices'].append(voice_emb)

    identities = list(identity_map.keys())
    genuine_scores  = []
    impostor_scores = []

    print("Building genuine and impostor pairs...")
    for label in identities:
        faces  = identity_map[label]['faces']
        voices = identity_map[label]['voices']

        # genuine pairs — same identity
        for face in faces[:5]:
            for voice in voices[:5]:
                score = F.cosine_similarity(face, voice).item()
                genuine_scores.append(score)

        # impostor pairs — different identity
        other_labels = [l for l in identities if l != label]
        for other_label in random.sample(other_labels, min(5, len(other_labels))):
            other_voices = identity_map[other_label]['voices']
            for face in faces[:2]:
                voice = random.choice(other_voices)
                score = F.cosine_similarity(face, voice).item()
                impostor_scores.append(score)

    eer = compute_eer(genuine_scores, impostor_scores)

    print(f"\n--- VERIFICATION RESULTS (Known Identity Pairs) ---")
    print(f"Genuine pairs:              {len(genuine_scores)}")
    print(f"Impostor pairs:             {len(impostor_scores)}")
    print(f"Genuine avg similarity:     {sum(genuine_scores)/len(genuine_scores):.4f}")
    print(f"Impostor avg similarity:    {sum(impostor_scores)/len(impostor_scores):.4f}")
    print(f"Separation gap:             {(sum(genuine_scores)/len(genuine_scores)) - (sum(impostor_scores)/len(impostor_scores)):.4f}")
    print(f"EER:                        {eer*100:.2f}%")
    print(f"(Lower EER = better. FAME 2024 best teams achieved ~20% EER)")

if __name__ == "__main__":
    random.seed(42)
    test_verification()