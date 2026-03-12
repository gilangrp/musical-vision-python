import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time

# Konfigurasi MediaPipe Tasks
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Audio setup - Ditambah sampai nada tinggi
samplerate = 44100
# 1-5 (Kiri), 6-10 (Kanan/Total)
notes = {
    1: 261.63, # C
    2: 293.66, # D
    3: 329.63, # E
    4: 349.23, # F
    5: 392.00, # G
    6: 440.00, # A
    7: 493.88, # B
    8: 523.25, # C Tinggi
    9: 587.33, # D Tinggi
    10: 659.25 # E Tinggi
}
last_play = 0
cooldown = 0.4

def play_tone(freq):
    duration = 0.3
    t = np.linspace(0, duration, int(samplerate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    sd.play(wave, samplerate)

def count_fingers_per_hand(hand_landmarks, handedness):
    fingers = 0
    tips_ids = [8, 12, 16, 20]
    label = handedness[0].category_name # "Left" atau "Right"

    # 4 Jari (Telunjuk - Kelingking)
    for tip_id in tips_ids:
        if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
            fingers += 1
            
    # Jempol (Logika dibedakan kiri/kanan karena posisi X jempol berbeda)
    # Setelah flip, "Left" di layar adalah tangan kanan asli user (tergantung kamera)
    # Kita gunakan logika: jika ujung jempol lebih jauh dari telunjuk secara horizontal
    if label == "Left": # Tangan kiri di layar
        if hand_landmarks[4].x > hand_landmarks[3].x: fingers += 1
    else: # Tangan kanan di layar
        if hand_landmarks[4].x < hand_landmarks[3].x: fingers += 1
        
    return fingers

# Inisialisasi Landmarker (num_hands diubah jadi 2)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2, 
    min_hand_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("Dual Hand Synth Started! Gunakan 10 jari untuk nada tinggi.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        total_fingers = 0

        if detection_result.hand_landmarks:
            # Iterasi setiap tangan yang terdeteksi
            for i in range(len(detection_result.hand_landmarks)):
                hand_landmarks = detection_result.hand_landmarks[i]
                handedness = detection_result.handedness[i]
                
                # Hitung jari untuk tangan ini dan tambahkan ke total
                total_fingers += count_fingers_per_hand(hand_landmarks, handedness)
                
                # Visualisasi titik tangan
                for lm in hand_landmarks:
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

            # Logika Suara
            if total_fingers in notes:
                now = time.time()
                if now - last_play > cooldown:
                    play_tone(notes[total_fingers])
                    last_play = now

        cv2.putText(frame, f"Total Fingers: {total_fingers}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Multi-Hand Synth", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()