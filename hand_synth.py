import cv2
import mediapipe as mp
import numpy as np
import time
from synthesizer import Player, Synthesizer, Waveform

# --- MediaPipe configuration ---
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Audio Setup  ---
player = Player()
player.open_stream()
#  Use Oscillator 1 (Sawtooth) for a richer sound, and Oscillator 2 (Sine) for a smoother tone
synth = Synthesizer(
    osc1_waveform=Waveform.sawtooth, 
    osc1_volume=0.3, 
    use_osc2=True, 
    osc2_waveform=Waveform.square, 
    osc2_volume=1
)

# C4 ke E5 (10 fingers)
note_map = {
    1: "C4", 2: "D4", 3: "E4", 4: "F4", 5: "G4",
    6: "A4", 7: "B4", 8: "C5", 9: "D5", 10: "E5"
}

last_play = 0
cooldown = 0.3

def count_fingers_per_hand(hand_landmarks, handedness):
    fingers = 0
    tips_ids = [8, 12, 16, 20]
    label = handedness[0].category_name
    for tip_id in tips_ids:
        if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
            fingers += 1
    # Thumb logic
    if label == "Left": # left hand on screen
        if hand_landmarks[4].x > hand_landmarks[3].x: fingers += 1
    else: # right hand on screen
        if hand_landmarks[4].x < hand_landmarks[3].x: fingers += 1
    return fingers


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("Analog Synth Started! Gunakan 1-10 jari.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        
        total_fingers = 0
        if detection_result.hand_landmarks:
            for i in range(len(detection_result.hand_landmarks)):
                hand_lms = detection_result.hand_landmarks[i]
                handedness = detection_result.handedness[i]
                total_fingers += count_fingers_per_hand(hand_lms, handedness)
                
                # Visualize hand landmarks
                for lm in hand_lms:
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

        # Logic to play notes based on finger count
        if total_fingers in note_map:
            now = time.time()
            if now - last_play > cooldown:
                current_note = note_map[total_fingers]
                # slow down the attack for a smoother sound
                wave = synth.generate_constant_wave(current_note, 0.2)
                player.play_wave(wave)
                last_play = now

        cv2.putText(frame, f"Fingers: {total_fingers} | Note: {note_map.get(total_fingers, '-')}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Synthesizer Library Hand", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()