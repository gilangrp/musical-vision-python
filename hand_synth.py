import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Konfigurasi MediaPipe Tasks ---
# Pastikan kamu sudah mendownload model 'hand_landmarker.task' 
# dari dokumentasi MediaPipe dan simpan di folder yang sama.
model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Audio setup
samplerate = 44100
notes = {1: 261.63, 2: 293.66, 3: 329.63, 4: 349.23, 5: 392.00}
last_play = 0
cooldown = 0.4

def play_tone(freq):
    duration = 0.3
    t = np.linspace(0, duration, int(samplerate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t) # Amplitudo 0.5 agar tidak clipping
    sd.play(wave, samplerate)

def count_fingers(hand_landmarks):
    # MediaPipe Hand Landmarks: 
    # Index 8, 12, 16, 20 adalah ujung jari (Tips)
    # Index 6, 10, 14, 18 adalah sendi PIP (Proximal Interphalangeal)
    fingers = 0
    tips_ids = [8, 12, 16, 20]
    
    # 4 Jari (Telunjuk - Kelingking)
    for tip_id in tips_ids:
        if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
            fingers += 1
            
    # Jempol (Logika sederhana berdasarkan posisi X relatif terhadap IP joint)
    # Asumsi tangan kanan menghadap kamera (setelah flip)
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers += 1
        
    return fingers

# Inisialisasi Landmarker dengan Video Mode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Gunakan 'with' agar resource otomatis tertutup
with HandLandmarker.create_from_options(options) as landmarker:
    print("Hand Synth Started (MediaPipe Tasks)")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Video Mode butuh timestamp dalam milidetik
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Convert ke MediaPipe Image Object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Deteksi
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        finger_count = 0

        if detection_result.hand_landmarks:
            # Ambil landmark tangan pertama
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Hitung jari
            finger_count = count_fingers(hand_landmarks)
            
            # Play sound logic
            if finger_count in notes:
                now = time.time()
                if now - last_play > cooldown:
                    play_tone(notes[finger_count])
                    last_play = now

            # Visualisasi sederhana (Opsional: gunakan DrawingUtils untuk lebih lengkap)
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        cv2.putText(frame, f"Fingers: {finger_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Hand Synth - Tasks API", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()