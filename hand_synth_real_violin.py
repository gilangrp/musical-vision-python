import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# --- Konfigurasi MediaPipe ---
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Audio Setup (Pygame Samples) ---
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Load sample suara (Pastikan file .wav ada di folder yang sama atau sesuaikan path-nya)
# Jika tidak ada file, kode ini akan error. Kamu bisa ganti dengan path file kamu.
try:
    samples = {
        1: pygame.mixer.Sound("samples/G4.wav"),
        2: pygame.mixer.Sound("samples/A4.wav"),
        3: pygame.mixer.Sound("samples/B4.wav"),
        4: pygame.mixer.Sound("samples/C5.wav"),
        5: pygame.mixer.Sound("samples/D5.wav"),
        6: pygame.mixer.Sound("samples/E5.wav"),
        7: pygame.mixer.Sound("samples/F#5.wav"),
        8: pygame.mixer.Sound("samples/G5.wav"),
        9: pygame.mixer.Sound("samples/G5.wav"),
        10: pygame.mixer.Sound("samples/E5.wav"),
    }
except:
    print("Peringatan: File .wav tidak ditemukan di folder 'samples/'.")
    samples = {}

last_play = 0
current_finger_count = 0
cooldown = 0.3

def count_fingers_per_hand(hand_landmarks, handedness):
    fingers = 0
    tips_ids = [8, 12, 16, 20]
    label = handedness[0].category_name
    for tip_id in tips_ids:
        if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
            fingers += 1
    if (label == "Left" and hand_landmarks[4].x > hand_landmarks[3].x) or \
       (label == "Right" and hand_landmarks[4].x < hand_landmarks[3].x):
        fingers += 1
    return fingers

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("Sample-based Synth Started!")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), int(time.time() * 1000))
        
        total_fingers = 0
        if detection_result.hand_landmarks:
            for i in range(len(detection_result.hand_landmarks)):
                total_fingers += count_fingers_per_hand(detection_result.hand_landmarks[i], detection_result.handedness[i])

        # Logika Mainkan Sample
        if total_fingers in samples and total_fingers != current_finger_count:
            now = time.time()
            if now - last_play > cooldown:
                # --- PERUBAHAN DI SINI ---
                # pygame.mixer.stop()  # Hentikan semua suara yang sedang berjalan
                pygame.mixer.fadeout(150)
                
                samples[total_fingers].play()
                last_play = now
                current_finger_count = total_fingers
                
        elif total_fingers == 0:
            if current_finger_count != 0:
                pygame.mixer.stop() # Hentikan suara saat tangan mengepal/hilang
            current_finger_count = 0

        cv2.putText(frame, f"Fingers: {total_fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Sample String Synth", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()