import os
import glob
import random
import gradio as gr
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, needed for saving figures without a display
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F

def make_waveform_image(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(waveform.squeeze(0).numpy(), color="steelblue", linewidth=0.5)
    ax.set_title("Voice waveform")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()

    out_path = "temp_batch_waveform.png"
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path

def make_embedding_heatmap(face_emb, voice_emb):
    face_vec = face_emb.detach().cpu().numpy().reshape(1, -1)
    voice_vec = voice_emb.detach().cpu().numpy().reshape(1, -1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 3))

    axes[0].imshow(face_vec, aspect="auto", cmap="coolwarm")
    axes[0].set_title(f"Face embedding (512 values)")
    axes[0].set_yticks([])

    axes[1].imshow(voice_vec, aspect="auto", cmap="coolwarm")
    axes[1].set_title(f"Voice embedding (512 values, after translator)")
    axes[1].set_yticks([])

    fig.tight_layout()
    out_path = "temp_embeddings.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path

from demo import (
    load_models,
    extract_face_embedding,
    extract_voice_embedding,
    compute_score,
)

print("Loading models once at startup...")
face_net, voice_net, face_translator, voice_translator = load_models()
print("Ready.\n")

from sklearn.decomposition import PCA
import torch.nn.functional as F

REFERENCE_ROOT = r"E:\mavceleb_v1"
N_REFERENCE_IDENTITIES = 10

def build_reference_projection():
    print("Building reference 2D projection from a handful of identities...")
    faces_root = os.path.join(REFERENCE_ROOT, "faces")
    all_ids = sorted(os.listdir(faces_root))
    chosen_ids = random.sample(all_ids, min(N_REFERENCE_IDENTITIES, len(all_ids)))

    points = []
    labels = []
    kinds = []  # "face" or "voice"

    for ident in chosen_ids:
        face_dir = os.path.join(faces_root, ident, "English")
        voice_dir = os.path.join(REFERENCE_ROOT, "voices", ident, "Urdu")
        if not os.path.exists(face_dir) or not os.path.exists(voice_dir):
            continue
        face_files = glob.glob(os.path.join(face_dir, "*", "*.jpg"))
        voice_files = glob.glob(os.path.join(voice_dir, "*", "*.wav"))
        if not face_files or not voice_files:
            continue
        try:
            f_emb = extract_face_embedding(face_files[0], face_net, face_translator)
            v_emb = extract_voice_embedding(voice_files[0], voice_net, voice_translator)
        except Exception:
            continue

        points.append(f_emb.detach().cpu().numpy().squeeze())
        labels.append(ident)
        kinds.append("face")

        points.append(v_emb.detach().cpu().numpy().squeeze())
        labels.append(ident)
        kinds.append("voice")

    points = np.array(points)
    # center, same principle as your tsne_urdu.py's alignment step
    points_centered = points - points.mean(axis=0, keepdims=True)

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(points_centered)

    print(f"Reference projection built with {len(labels)} points from {len(set(labels))} identities.")
    return pca, points.mean(axis=0, keepdims=True), coords_2d, labels, kinds


REF_PCA, REF_MEAN, REF_COORDS, REF_LABELS, REF_KINDS = build_reference_projection()


def project_new_point(embedding_tensor):
    vec = embedding_tensor.detach().cpu().numpy().reshape(1, -1)
    vec_centered = vec - REF_MEAN
    return REF_PCA.transform(vec_centered)[0]


def make_projection_plot(face_emb, voice_emb, face_label="new face", voice_label="new voice"):
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (x, y) in enumerate(REF_COORDS):
        color = "lightblue" if REF_KINDS[i] == "face" else "lightgreen"
        marker = "o" if REF_KINDS[i] == "face" else "^"
        ax.scatter(x, y, c=color, marker=marker, s=40, alpha=0.6, edgecolors="gray")

    new_face_xy = project_new_point(face_emb)
    new_voice_xy = project_new_point(voice_emb)

    ax.scatter(*new_face_xy, c="red", marker="o", s=200, edgecolors="black",
               linewidths=2, label=face_label, zorder=5)
    ax.scatter(*new_voice_xy, c="blue", marker="^", s=200, edgecolors="black",
               linewidths=2, label=voice_label, zorder=5)
    ax.plot([new_face_xy[0], new_voice_xy[0]], [new_face_xy[1], new_voice_xy[1]],
            "k--", alpha=0.5, zorder=4)

    ax.set_title("Where this face and voice land in the shared embedding space\n"
                  "(gray = reference identities, red/blue = your uploaded pair)")
    ax.legend()
    fig.tight_layout()

    out_path = "temp_projection.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path

def run_single_comparison(image_path, audio_path):
    if image_path is None or audio_path is None:
        return "Please provide both a face image and a voice clip.", None, None

    try:
        face_emb = extract_face_embedding(image_path, face_net, face_translator)
        voice_emb = extract_voice_embedding(audio_path, voice_net, voice_translator)
        score = compute_score(face_emb, voice_emb)
        spectrogram_path = make_embedding_heatmap(face_emb, voice_emb)
        projection_path = make_projection_plot(face_emb, voice_emb)
    except Exception as e:
        return f"Error: {e}", None, None

    verdict = "Likely MATCH" if score > 0.5 else "Likely NOT a match"
    return f"Similarity score: {score:.4f}\n\n{verdict}", spectrogram_path, projection_path


def run_batch_evaluation(root_folder, num_samples, face_language):
    if not root_folder or not os.path.isdir(root_folder):
        yield [], "Please provide a valid dataset root folder.", "", None, None
        return

    faces_root = os.path.join(root_folder, "faces")
    voices_root = os.path.join(root_folder, "voices")
    if not os.path.exists(faces_root) or not os.path.exists(voices_root):
        yield [], f"Expected 'faces' and 'voices' subfolders inside {root_folder}", "", None, None
        return

    all_identities = sorted(os.listdir(faces_root))
    n_to_sample = min(int(num_samples), len(all_identities))
    identities = random.sample(all_identities, n_to_sample)

    face_embs = {}
    voice_embs = {}
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    yield [], "", log(f"Selected {n_to_sample} random identities: {', '.join(identities)}\n"), None, None

    for ident in identities:
        if face_language == "Urdu (matches thesis protocol)":
            face_lang = "Urdu"
        elif face_language == "English":
            face_lang = "English"
        else:
            face_lang = random.choice(["English", "Urdu"])

        face_dir = os.path.join(faces_root, ident, face_lang)
        voice_dir = os.path.join(voices_root, ident, "Urdu")

        if not os.path.exists(face_dir) or not os.path.exists(voice_dir):
            yield [], "", log(f"[{ident}] SKIPPED — missing {face_lang} face or Urdu voice folder"), None, None
            continue

        face_files = glob.glob(os.path.join(face_dir, "*", "*.jpg"))
        voice_files = glob.glob(os.path.join(voice_dir, "*", "*.wav"))

        if not face_files or not voice_files:
            yield [], "", log(f"[{ident}] SKIPPED — no files found"), None, None
            continue

        chosen_face = random.choice(face_files)
        chosen_voice = random.choice(voice_files)

        waveform_img = make_waveform_image(chosen_voice)

        yield ([], "", log(f"[{ident}] face: {face_lang}/{os.path.basename(os.path.dirname(chosen_face))}/{os.path.basename(chosen_face)}  |  "
                            f"voice: Urdu/{os.path.basename(os.path.dirname(chosen_voice))}/{os.path.basename(chosen_voice)}"),
               chosen_face, waveform_img)

        try:
            face_embs[ident] = extract_face_embedding(chosen_face, face_net, face_translator)
            voice_embs[ident] = extract_voice_embedding(chosen_voice, voice_net, voice_translator)
        except Exception as e:
            yield [], "", log(f"[{ident}] ERROR extracting embeddings: {e}"), None, None
            continue

    valid_ids = list(face_embs.keys())
    n = len(valid_ids)
    if n < 2:
        yield [], f"Only {n} valid identities extracted — need at least 2.", log("\nStopped: not enough valid identities."), None, None
        return

    yield [], "", log(f"\nExtracted {n} valid identities. Running Rank-1 retrieval across all pairs...\n"), None, None

    rows = []
    correct = 0

    for face_id in valid_ids:
        best_score = -2.0
        best_match = None

        for voice_id in valid_ids:
            score = compute_score(face_embs[face_id], voice_embs[voice_id])
            if score > best_score:
                best_score = score
                best_match = voice_id

        is_correct = (best_match == face_id)
        if is_correct:
            correct += 1

        rows.append([face_id, best_match, f"{best_score:.4f}", "YES" if is_correct else "NO"])
        yield rows, "", log(f"[{face_id}] best match -> {best_match} (score {best_score:.4f}) "
                              f"{'CORRECT' if is_correct else 'WRONG'}"), None, None

    accuracy = correct / n * 100
    random_chance = 100 / n
    summary = (f"Rank-1 accuracy: {correct}/{n} correct = {accuracy:.2f}%\n"
               f"Random chance baseline: {random_chance:.2f}%")

    yield rows, summary, log(f"\nDone. {summary}"), None, None


with gr.Blocks(title="Cross-Modal Face-Voice Matching Demo",
                theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo_ui:
    gr.Markdown("# Cross-Modal Face-Voice Biometric Matching")
    gr.Markdown("Live demo of the trained ArcFace + ECAPA-TDNN pipeline.")

    with gr.Tab("Single Comparison"):
        gr.Markdown("Upload a face image and a voice clip to see the similarity score.")
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Face Image")
            audio_input = gr.Audio(type="filepath", label="Voice Clip")
        compare_btn = gr.Button("Compare", variant="primary")
        result_output = gr.Textbox(label="Result", lines=3)
        with gr.Row():
            spectrogram_output = gr.Image(label="Face & Voice Embeddings")
            projection_output = gr.Image(label="Embedding Space Projection")
        compare_btn.click(
            fn=run_single_comparison,
            inputs=[image_input, audio_input],
            outputs=[result_output, spectrogram_output, projection_output],
        )

    with gr.Tab("Batch Evaluation"):
        gr.Markdown("Point at a MAVCeleb-style dataset root (containing `faces/` and `voices/`) "
                     "to run Rank-1 retrieval evaluation across multiple identities.")
        folder_input = gr.Textbox(label="Dataset root folder path",
                                    placeholder=r"E:\mavceleb_v1")
        n_samples = gr.Slider(minimum=5, maximum=64, value=15, step=1,
                                label="Number of identities to test")
        face_language = gr.Radio(
            choices=["Urdu (matches thesis protocol)", "English", "Mixed (random per identity)"],
            value="Urdu (matches thesis protocol)",
            label="Face language source"
        )
        run_batch_btn = gr.Button("Run Batch Evaluation", variant="primary")
        with gr.Row():
            current_face = gr.Image(label="Currently Processing: Face")
            current_waveform = gr.Image(label="Currently Processing: Voice Waveform")
        batch_table = gr.Dataframe(
            headers=["Identity", "Predicted Match", "Score", "Correct?"],
            label="Results"
        )
        batch_summary = gr.Textbox(label="Summary")
        live_log = gr.Textbox(label="Live Log", lines=15, autoscroll=False)
        run_batch_btn.click(
            fn=run_batch_evaluation,
            inputs=[folder_input, n_samples, face_language],
            outputs=[batch_table, batch_summary, live_log, current_face, current_waveform],
        )

if __name__ == "__main__":
    demo_ui.launch(allowed_paths=[r"E:\mavceleb_v1"])