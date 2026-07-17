import os
import cv2
import subprocess
import yt_dlp

CORPUS_DIR = r"build_youtube_corpus"
FACE_DIR = os.path.join(CORPUS_DIR, "faces", "English")
VOICE_DIR = os.path.join(CORPUS_DIR, "voices", "English")
# "English" here is just a folder label to mirror MAVCeleb's language-subfolder convention,
# not a claim that the speech is actually English — these are Romanian speakers,
# this naming is purely for structural compatibility with the existing dataset/loader conventions

YOUTUBE_TARGETS = {
    # Men
    "CristianPresura": "https://www.youtube.com/watch?v=ZPu-EbA5f2I",
    "GheorgheBuhnici": "https://www.youtube.com/watch?v=z-VQi0iGpDk",
    "SpiritulVremii": "https://www.youtube.com/watch?v=yG-q7_fmxec",
    "Zaiafet": "https://www.youtube.com/watch?v=n09AI9bui2I",
    "DorianPopa": "https://www.youtube.com/watch?v=92UzrkZGmWw",

    # Women
    "IrinaManea": "https://www.youtube.com/watch?v=wxk1ahooRWg",
    "Mimi": "https://www.youtube.com/watch?v=_emLikcsV-U",
    "Marilu": "https://www.youtube.com/watch?v=F63RARBLQOI",
    "LorenaVisan": "https://www.youtube.com/watch?v=d3WwTdikN-4&t=17s",
    "RuxSiOpulenta": "https://www.youtube.com/watch?v=3LB1RdxQg0A",
    # Men — Stand-up (new)
    "RaduBucalae":    "https://www.youtube.com/watch?v=67r0QEMIb1g",
    "CatalinBordea":  "https://www.youtube.com/watch?v=zH-8uhhnWc0",
    "Teo":            "https://www.youtube.com/watch?v=nC6MXCMJ5PQ",
    "CosminNatanticu":"https://www.youtube.com/watch?v=sEMwLLthMqs",
    "Micutzu":        "https://www.youtube.com/watch?v=2CrUHESag7U",
    "DanBadea":       "https://www.youtube.com/watch?v=49tqnbMOUR4",
    "ClaudiuPopa":    "https://www.youtube.com/watch?v=bXZLrV3WLp0",
    "Vio":            "https://www.youtube.com/watch?v=A3PFZE1YXyc",
    "MihaiBobonete":  "https://www.youtube.com/watch?v=nQBMnoupeNQ",
    "NelucCortea":    "https://www.youtube.com/watch?v=IKvYZoCAYB4",
    "AlexMocanu":     "https://www.youtube.com/watch?v=MRY4CopDU_A",
    "AndreiCiobanu":  "https://www.youtube.com/watch?v=bzqoWcr1EFI",
    # Men — Vloggers / Entertainers (new)
    "Selly":          "https://www.youtube.com/watch?v=boCyTnKCnqM",
    "MirceaBravo":    "https://www.youtube.com/watch?v=0jpDgZ1umfw",
    "AlexVelea":      "https://www.youtube.com/watch?v=SwHbnZp3b3c",
    "Speak":          "https://www.youtube.com/watch?v=uaR9aYTGrPY",
    "BogdanTeches":   "https://www.youtube.com/watch?v=5QFsr9xxJFQ",
    "MihaiGainusa":   "https://www.youtube.com/watch?v=d8w9rl-2FPc",
    "RazvanGogan":    "https://www.youtube.com/watch?v=hC4oHSjx020",
    "VladGherman":    "https://www.youtube.com/watch?v=HMUs9S3jm_Y",
    "CristianFlorea": "https://www.youtube.com/watch?v=PUzAjSdOfug",
    "DaniOtil":       "https://www.youtube.com/watch?v=UQyii9ObPdI",
    "VladMunteanu":   "https://www.youtube.com/watch?v=5Xn9kn8NGEI",
    "MihaiMorar":     "https://www.youtube.com/watch?v=dRGAX7yBsXc",
    "PavelBartos":    "https://www.youtube.com/watch?v=OQVY8i3zSbk",
    # Women — Stand-up (new)
    "MariaPopovici":  "https://www.youtube.com/watch?v=1d_-qcrowxU",
    "IoanaState":     "https://www.youtube.com/watch?v=vTryt3DgePQ",
    # Women — Vloggers / Influencers (new)
    "AndraGogan":     "https://www.youtube.com/watch?v=9SxBOR2d8DU",
    "AndreeaBalaban": "https://www.youtube.com/watch?v=fnG6mGu-qYk",
    "IoanaGrama":     "https://www.youtube.com/watch?v=djvxFsRbNVY",
    "IrinaDeaconescu":"https://www.youtube.com/watch?v=6gMgVHGfZt4",
    "SanzianaNegru":  "https://www.youtube.com/watch?v=3XKR-BqNGy0",
    "CarmenGrebenisan":"https://www.youtube.com/watch?v=nWXmcBMb5sg",
    "IrinaColumbeanu":"https://www.youtube.com/watch?v=mqSDm2eNsQ0",
    "JamilaCuisine":  "https://www.youtube.com/watch?v=E0hkLMCHpDg",
    "CristinaAlmasan":"https://www.youtube.com/watch?v=8wv2wFzV76U",
    "CezaraCristescu":"https://www.youtube.com/watch?v=zcfhm-8Q_KI",
    "BiancaAdam":     "https://www.youtube.com/watch?v=79KyCOOpWRQ",
    "AnaMorodan":     "https://www.youtube.com/watch?v=FyPWQloYEuU",
    "AlinaCeusan":    "https://www.youtube.com/watch?v=GtuTeXLUoC8",
}
# ~50 named public figures here at the source, spanning multiple content categories
# (stand-up comedians, vloggers, influencers), roughly gender balanced by construction
# this is the "50 identities as collected" number — downstream, some get dropped for missing/bad data,
# which is where the 41-identity evaluation number comes from later

def setup_directories():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(VOICE_DIR, exist_ok=True)
    os.makedirs("temp", exist_ok=True)

def get_identity_id_map():
    """Maps each real name to an anonymized ID001, ID002, ... identifier,
    in the fixed dict-insertion order of YOUTUBE_TARGETS, so the mapping
    is stable and reproducible across runs."""
    return {
        name: f"ID{str(i + 1).zfill(3)}"
        for i, name in enumerate(YOUTUBE_TARGETS.keys())
    }
    # anonymization here is purely nominal — these are public figures, real names are
    # kept in a separate mapping file, but the ID scheme mirrors MAVCeleb's idXXX convention
    # for structural consistency with the rest of the pipeline, not for privacy in a strict sense

def extract_multiple_faces(video_path, identity, video_id, save_dir, num_faces=10):
    """Scans the video and extracts N distinct faces spaced temporally.
    Saves into save_dir/identity/video_id/face_N.jpg, mirroring MAVCeleb's
    id/video_id/frames nesting convention."""
    print(f"   -> Scanning for {num_faces} distinct facial frames...")
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # this IS the Haar cascade detector referenced in the thesis — confirmed here directly,
    # lightweight classical CV, not a deep detector, chosen because its fast and good enough
    # for frontal, well-lit YouTube video frames — this is a data curation step, not part of the trained model

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    #skip the first 10 seconds
    start_frame = int(10 * fps)
    # avoids channel intros/titles, same logic as the audio center-crop later —
    # both are working around the fact that YouTube videos dont start with clean content
    if total_frames <= start_frame:
        start_frame = 0 # fallback just in case the video is super short

    # step size to jump through the video in num_faces equal chunks
    remaining_frames = total_frames - start_frame
    step = int(remaining_frames / num_faces)
    # this spaces the 10 extracted faces evenly across the video's remaining duration,
    # so youre not just grabbing 10 near-duplicate frames from the same 2-second window

    identity_session_dir = os.path.join(save_dir, identity, video_id)
    os.makedirs(identity_session_dir, exist_ok=True)

    faces_saved = 0

    for i in range(num_faces):
        target_frame = start_frame + (i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        face_found = False
        attempts = 0

        # Scan forward from the target jump point until we find a clear face (max 150 frames)
        while cap.isOpened() and not face_found and attempts < 150:
            # this is the retry logic — if the exact target frame doesnt have a clean detectable face
            # (blink, turned head, cut to b-roll), it keeps advancing frame by frame until one is found
            # or gives up after 150 frames (~5-6 seconds at typical fps)
            ret, frame = cap.read()
            if not ret: break
            attempts += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
            # minSize=(100,100) filters out small/background faces, only accepts reasonably close-up ones

            if len(faces) > 0:
                # Grab the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                # if multiple faces are in frame (interviewer, audience member), this always picks
                # the biggest bounding box, a reasonable heuristic for "the main subject"

                # 20% margin for the VGG jawline
                margin = int(w * 0.2)
                # comment says "VGG" here, leftover naming from before the ArcFace swap —
                # the margin logic itself is still valid, just the comment wasnt updated
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)

                face_crop = frame[y1:y2, x1:x2]

                # Save as face_0.jpg, face_1.jpg, etc. within identity/video_id/
                save_path = os.path.join(identity_session_dir, f"face_{faces_saved}.jpg")
                cv2.imwrite(save_path, face_crop)
                faces_saved += 1
                face_found = True
                print(f"      Face {faces_saved}/{num_faces} saved.")

    cap.release()
    if faces_saved < num_faces:
        print(f"   -> WARNING: Only found {faces_saved}/{num_faces} clean faces in this video.")
        # this warning is printed but not logged anywhere persistent — if an identity ends up
        # with very few face crops, this is the only place that would have told you why

def get_audio_duration(path):
    """Returns duration in seconds via ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())
    # shells out to ffprobe rather than using a python audio library — simple, reliable,
    # matches the ffmpeg-based approach used everywhere else in this file

def process_audio(video_path, save_path):
    """Extracts the full audio track, mono, 16kHz."""
    print(f"   -> Extracting and downsampling acoustic waveform...")
    command = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn', '-ac', '1', '-ar', '16000', save_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # -vn strips video, -ac 1 forces mono, -ar 16000 matches ECAPA-TDNN's expected sample rate exactly,
    # same 16kHz convention used everywhere else in the pipeline (extract_features_all.py, demo.py)
    # stderr/stdout suppressed — if ffmpeg fails silently here, youd only notice downstream when the file is missing

def extract_center_crop(full_wav_path, crop_save_path, crop_seconds=3.0):
    """
    Extracts a fixed-length segment centered on the midpoint of the full
    track. Center cropping is preferred over cropping from the start
    because YouTube videos frequently open with music, applause, or crowd
    noise before the speaker begins, making the midpoint a more reliable
    source of clean speech than the start of the recording.
    """
    print(f"   -> Extracting {crop_seconds}s center crop...")
    duration = get_audio_duration(full_wav_path)

    if duration <= crop_seconds:
        # track is shorter than the crop window, just use the whole thing
        start = 0.0
        actual_crop = duration
    else:
        center = duration / 2.0
        start = max(0.0, center - crop_seconds / 2.0)
        actual_crop = crop_seconds
        # same 3-second convention as extract_features_all.py's voice processing,
        # but different cropping STRATEGY — that script does a deterministic center crop
        # of an already-loaded waveform in memory, this one does it via ffmpeg on the file directly
        # both are "center crop" in spirit, just implemented at different pipeline stages

    command = [
        'ffmpeg', '-y',
        '-i', full_wav_path,
        '-ss', str(start), '-t', str(actual_crop),
        '-ac', '1', '-ar', '16000',
        crop_save_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    setup_directories()
    id_map = get_identity_id_map()

    # Save the name-to-ID mapping once, kept separately and not committed
    # to any public repo folder, purely for internal bookkeeping.
    mapping_path = os.path.join(CORPUS_DIR, "identity_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as f:
        for name, anon_id in id_map.items():
            f.write(f"{anon_id}\t{name}\n")

    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best',
        'outtmpl': 'temp/%(id)s.%(ext)s',
        'quiet': True
    }
    # caps at 720p — reasonable, faces dont need higher resolution than that for detection/embedding purposes,
    # keeps download size/time down across ~50 videos

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for name, url in YOUTUBE_TARGETS.items():
            anon_id = id_map[name]
            print(f"\nProcessing Identity: [{anon_id}]")

            try:
                print(f"   -> Downloading media stream...")
                info = ydl.extract_info(url, download=True)
                video_path = f"temp/{info['id']}.mp4"
                video_id = info['id']

                identity_voice_dir = os.path.join(VOICE_DIR, anon_id, video_id)
                os.makedirs(identity_voice_dir, exist_ok=True)

                voice_save_path = os.path.join(identity_voice_dir, "voice_full.wav")
                voice_crop_path = os.path.join(identity_voice_dir, "voice_3s.wav")

                # Simple check if audio already exists to avoid re-downloading
                if os.path.exists(voice_save_path):
                    print("   -> Audio data already exists. Skipping.")
                    continue
                    # this makes the script resumable/idempotent — if it crashes partway through
                    # the ~50 identities, re-running it wont redo work already done
                    # NOTE: this check only guards audio, not faces — if faces failed but audio succeeded,
                    # re-running would still skip this identity entirely and not retry the face extraction

                extract_multiple_faces(video_path, anon_id, video_id, FACE_DIR)
                process_audio(video_path, voice_save_path)
                extract_center_crop(voice_save_path, voice_crop_path, crop_seconds=3.0)

                os.remove(video_path)
                # deletes the raw downloaded video after processing — keeps disk usage bounded
                # since only faces + audio are actually needed downstream, not the original video

            except Exception as e:
                print(f"   -> FAILED to process {anon_id} ({name}): {e}")
                # this except is NOT silent like extract_features_all.py's — it prints which identity
                # and the actual exception. this is likely the traceable source of your "9 empty folders"
                # mentioned in the thesis (test_in_the_wild.py label-compression) — a download or detection
                # failure here for a given identity would leave that identity's folder empty or missing

    print("\nCorpus generation complete.")

if __name__ == "__main__":
    main()