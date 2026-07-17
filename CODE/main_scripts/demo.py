"""
demo.py — Live face-voice matching demo.
Run examples:
    python demo.py --face path/to/face.jpg --voice path/to/voice.wav
"""

import argparse
import torch
import torch.nn.functional as F
import torchaudio
import cv2

from models import FaceEncoder, VoiceEncoder, ModalityTranslator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\model_arcface.pth"


def load_models():
    print("Loading encoders and translators...")
    face_net = FaceEncoder().eval()
    voice_net = VoiceEncoder().eval()

    face_translator = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    face_translator.eval()
    voice_translator.eval()

    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})\n")
    return face_net, voice_net, face_translator, voice_translator


def extract_face_embedding(image_path, face_net, face_translator):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = face_net.app.get(img_bgr)
    if len(faces) == 0:
        raise ValueError(f"No face detected in: {image_path}")

    raw_emb = torch.tensor(faces[0].embedding, dtype=torch.float32).unsqueeze(0)
    raw_emb = F.normalize(raw_emb, p=2, dim=1).to(DEVICE)

    with torch.no_grad():
        translated = face_translator(raw_emb)
    return translated


def extract_voice_embedding(audio_path, voice_net, voice_translator):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    target_len = 16000 * 3
    if waveform.shape[1] < target_len:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
    else:
        start = (waveform.shape[1] - target_len) // 2
        waveform = waveform[:, start:start + target_len]

    waveform_input = waveform.squeeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_emb = voice_net(waveform_input)
        translated = voice_translator(raw_emb)
    return translated


def compute_score(face_emb, voice_emb):
    return F.cosine_similarity(face_emb, voice_emb).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", required=True, help="Path to a face image")
    parser.add_argument("--voice", required=True, help="Path to a voice clip")
    args = parser.parse_args()

    face_net, voice_net, face_translator, voice_translator = load_models()

    print(f"Extracting face embedding from: {args.face}")
    face_emb = extract_face_embedding(args.face, face_net, face_translator)

    print(f"Extracting voice embedding from: {args.voice}")
    voice_emb = extract_voice_embedding(args.voice, voice_net, voice_translator)

    score = compute_score(face_emb, voice_emb)
    print(f"\nSimilarity score: {score:.4f}")