import os
import cv2
import subprocess
import yt_dlp

CORPUS_DIR = r"build_youtube_corpus"
FACE_DIR = os.path.join(CORPUS_DIR, "faces", "English")
VOICE_DIR = os.path.join(CORPUS_DIR, "voices", "English")


YOUTUBE_TARGETS = {
    # Men
    "CristianPresura": "https://www.youtube.com/watch?v=ZPu-EbA5f2I",
    "GheorgheBuhnici": "https://www.youtube.com/watch?v=z-VQi0iGpDk",
    "SpiritulVremii": "https://www.youtube.com/watch?v=yG-q7_fmxec"
    "Zaiafet:" "https://www.youtube.com/watch?v=n09AI9bui2I",
    "DorianPopa": "https://www.youtube.com/watch?v=92UzrkZGmWw",
    
    # Women
    "IrinaManea": "https://www.youtube.com/watch?v=wxk1ahooRWg",
    "Mimi": "https://www.youtube.com/watch?v=_emLikcsV-U",
    "Marilu": "https://www.youtube.com/watch?v=F63RARBLQOI",
    "LorenaVisan": "https://www.youtube.com/watch?v=d3WwTdikN-4&t=17s",
    "RuxSiOpulenta": "https://www.youtube.com/watch?v=3LB1RdxQg0A"
}

def setup_directories():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(VOICE_DIR, exist_ok=True)
    os.makedirs("temp", exist_ok=True)

def extract_multiple_faces(video_path, identity, save_dir, num_faces=5):
    """Scans the video and extracts N distinct faces spaced temporally."""
    print(f"   -> Scanning for {num_faces} distinct facial frames...")
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    #skip the first 10 seconds
    start_frame = int(10 * fps)
    if total_frames <= start_frame:
        start_frame = 0 # fallback just in case the video is super short

    # step size to jump through the video in 5 equal chunks
    remaining_frames = total_frames - start_frame
    step = int(remaining_frames / num_faces)

    faces_saved = 0

    for i in range(num_faces):
        target_frame = start_frame + (i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        face_found = False
        attempts = 0
        
        # Scan forward from the target jump point until we find a clear face (max 150 frames)
        while cap.isOpened() and not face_found and attempts < 150:
            ret, frame = cap.read()
            if not ret: break
            attempts += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                # Grab the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]

                # 20% margin for the VGG jawline
                margin = int(w * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)

                face_crop = frame[y1:y2, x1:x2]
                
                # Save as Identity_1.jpg, Identity_2.jpg, etc.
                save_path = os.path.join(save_dir, f"{identity}_{faces_saved + 1}.jpg")
                cv2.imwrite(save_path, face_crop)
                faces_saved += 1
                face_found = True
                print(f"      Face {faces_saved}/{num_faces} saved.")

    cap.release()
    if faces_saved < num_faces:
        print(f"   -> WARNING: Only found {faces_saved}/{num_faces} clean faces in this video.")

def process_audio(video_path, save_path):
    print(f"   -> Extracting and downsampling acoustic waveform...")
    command = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-vn', '-ac', '1', '-ar', '16000', save_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    setup_directories()
    
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best',
        'outtmpl': 'temp/%(id)s.%(ext)s',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for identity, url in YOUTUBE_TARGETS.items():
            print(f"\nProcessing Identity: [{identity}]")
            
            voice_save_path = os.path.join(VOICE_DIR, f"{identity}.wav")
            
            # Simple check if audio already exists to avoid re-downloading
            if os.path.exists(voice_save_path):
                print("   -> Audio data already exists. Assuming faces are done. Skipping.")
                continue

            try:
                print(f"   -> Downloading media stream...")
                info = ydl.extract_info(url, download=True)
                video_path = f"temp/{info['id']}.mp4"
                
                extract_multiple_faces(video_path, identity, FACE_DIR)
                process_audio(video_path, voice_save_path)
                
                os.remove(video_path)
                
            except Exception as e:
                print(f"   -> FAILED to process {identity}: {e}")

    print("\nCorpus generation complete.")

if __name__ == "__main__":
    main()