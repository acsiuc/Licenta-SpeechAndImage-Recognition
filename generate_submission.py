import torch
import torch.nn.functional as F
import torchaudio
import os
from models import FaceEncoder, VoiceEncoder, ModalityTranslator
from torchvision import transforms
from PIL import Image

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"C:\Users\Axiuc\OneDrive - Technical University of Cluj-Napoca\Desktop\Licenta\mav_celeb_v1_test"
CKPT     = r"model_cu_transformer.pth"

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_models():
    face_net  = FaceEncoder().to(DEVICE).eval()
    voice_net = VoiceEncoder().to(DEVICE).eval()
    face_translator  = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    face_translator.load_state_dict(ckpt['face_translator'])
    voice_translator.load_state_dict(ckpt['voice_translator'])
    face_translator.eval()
    voice_translator.eval()
    return face_net, voice_net, face_translator, voice_translator

def process_pair(face_path, voice_path, face_net, voice_net, face_translator, voice_translator):
    img = Image.open(face_path).convert("RGB")
    face_tensor = face_transform(img).unsqueeze(0).to(DEVICE)
    face_emb = face_net(face_tensor)
    face_proj = face_translator(face_emb)

    waveform, sr = torchaudio.load(voice_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    target_len = 16000 * 3
    if waveform.shape[1] < target_len:
        import torch.nn.functional as F_
        waveform = F_.pad(waveform, (0, target_len - waveform.shape[1]))
    else:
        start = (waveform.shape[1] - target_len) // 2
        waveform = waveform[:, start: start + target_len]

    waveform_input = waveform.squeeze(0).unsqueeze(0).to(DEVICE)
    voice_emb  = voice_net(waveform_input)
    voice_proj = voice_translator(voice_emb)

    # L2 distance as required by the challenge
    score = torch.norm(face_proj - voice_proj).item()
    return score

def generate_scores(pairs_file, face_lang, voice_lang, output_file,
                    face_net, voice_net, face_translator, voice_translator):
    with open(pairs_file, 'r') as f:
        lines = f.read().strip().split('\n')

    print(f"Generating scores for {len(lines)} pairs -> {output_file}")
    results = []

    with torch.no_grad():
        for i, line in enumerate(lines):
            parts      = line.strip().split()
            pair_id    = parts[0]
            voice_path = os.path.join(TEST_DIR, parts[1])
            face_path  = os.path.join(TEST_DIR, parts[2])
            score      = process_pair(face_path, voice_path,
                                      face_net, voice_net,
                                      face_translator, voice_translator)
            results.append(f"{pair_id} {score:.4f}")
            if i % 100 == 0:
                print(f"  {i}/{len(lines)} done...")

    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"Saved {output_file}")

if __name__ == "__main__":
    face_net, voice_net, face_translator, voice_translator = load_models()

    # English heard — trained on English, tested on English
    generate_scores(
        pairs_file   = os.path.join(TEST_DIR, "English_test.txt"),
        face_lang    = "English",
        voice_lang   = "English",
        output_file  = "sub_score_English_heard.txt",
        face_net=face_net, voice_net=voice_net,
        face_translator=face_translator, voice_translator=voice_translator
    )

    # Urdu unheard — trained on English, tested on Urdu
    generate_scores(
        pairs_file   = os.path.join(TEST_DIR, "Urdu_test.txt"),
        face_lang    = "Urdu",
        voice_lang   = "Urdu",
        output_file  = "sub_score_Urdu_unheard.txt",
        face_net=face_net, voice_net=voice_net,
        face_translator=face_translator, voice_translator=voice_translator
    )

    print("\nDone. Now zip and submit:")
    print("Need 4 files total — English heard, English unheard, Urdu heard, Urdu unheard")
    print("You only have 2 test files so submit what you have.")