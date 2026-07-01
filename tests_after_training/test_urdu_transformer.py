import torch
import torch.nn.functional as F
import os
import glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ModalityTranslator, TransformerCrossAttention, JointClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\Downloads\urdu_test_embeddings"
NUM_CLASSES = 198  # must match the classifier's trained output size


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


def test_fusion_verification():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    print(f"Loading {len(files)} Urdu test samples...")

    # load all four trained components — this is the difference from
    # test_urdu.py, which only loads the two translators
    face_translator = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)
    transformer_fusion = TransformerCrossAttention(embed_dim=512).to(DEVICE)
    classifier = JointClassifier(num_classes=NUM_CLASSES, embedding_dim=512).to(DEVICE)

    checkpoint = torch.load("model_arcface.pth", map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    transformer_fusion.load_state_dict(checkpoint['transformer_fusion'])
    classifier.load_state_dict(checkpoint['classifier'])

    face_translator.eval()
    voice_translator.eval()
    transformer_fusion.eval()
    classifier.eval()

    # load raw embeddings and true identity labels for all 64 test identities
    faces, voices, labels = [], [], []
    with torch.no_grad():
        for f in files:
            data = torch.load(f, weights_only=False)
            faces.append(data['face_emb'].squeeze(0).to(DEVICE))
            voices.append(data['voice_emb'].squeeze(0).to(DEVICE))
            labels.append(data['label'].item())

    n = len(faces)

    # sanity check: confirm labels are the original 0-197 training identity
    # indices, not a locally re-numbered 0-63 range, before trusting the
    # classifier lookup below
    print(f"First 10 labels: {labels[:10]}")
    print(f"Label range: min = {min(labels)}, max = {max(labels)}")
    print(f"Unique labels: {len(set(labels))} (should equal {n})\n")

    print(f"Building {n} genuine pairs and {n * (n - 1)} impostor pairs...")

    genuine_scores = []
    impostor_scores = []

    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                face_emb = faces[i].unsqueeze(0)
                voice_emb = voices[j].unsqueeze(0)

                f_512 = face_translator(face_emb)
                v_512 = voice_translator(voice_emb)

                fused = transformer_fusion(f_512, v_512)
                logits = classifier(fused)
                probs = F.softmax(logits, dim=1)

                # score = classifier's confidence that this fused pair
                # belongs to the identity of the FACE half of the pair.
                # For a genuine pair (i == j) this is the correct-identity
                # confidence. For an impostor pair (i != j) this measures
                # whether the classifier still (wrongly) assigns high
                # confidence to the face's true identity despite the
                # mismatched voice.
                true_identity = labels[i]
                score = probs[0, true_identity].item()

                if i == j:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

    eer = compute_eer(genuine_scores, impostor_scores)

    avg_genuine = sum(genuine_scores) / len(genuine_scores)
    avg_impostor = sum(impostor_scores) / len(impostor_scores)

    print(f"\n--- PROTOCOL B: FUSION + CLASSIFIER VERIFICATION (Urdu Test Set) ---")
    print(f"Total identities:              {n}")
    print(f"Genuine pairs:                 {len(genuine_scores)}")
    print(f"Impostor pairs:                {len(impostor_scores)}")
    print(f"Avg genuine confidence:        {avg_genuine:.4f}")
    print(f"Avg impostor confidence:       {avg_impostor:.4f}")
    print(f"Separation gap:                {avg_genuine - avg_impostor:.4f}")
    print(f"EER:                           {eer * 100:.2f}%")
    print(f"\n(Compare against Protocol A / test_urdu.py cosine-similarity EER: 9.10%)")
    print("Note: the classifier was trained exclusively on genuine pairs during")
    print("train.py — its behavior on impostor pairs here is an out-of-distribution")
    print("generalization test, not a directly optimized objective.")


if __name__ == "__main__":
    test_fusion_verification()
