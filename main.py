import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time

# Hand detection setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Audio settings
samplerate = 44100

# Musical notes frequencies
notes = {
    1: 261.63,  # C
    2: 293.66,  # D
    3: 329.63,  # E
    4: 349.23,  # F
    5: 392.00   # G
}

last_play = 0
cooldown = 0.4


def play(freq):
    duration = 0.3
    t = np.linspace(0, duration, int(samplerate * duration), False)

    wave = np.sin(2 * np.pi * freq * t)

    sd.play(wave, samplerate, blocking=False)


def count_fingers(hand_landmarks):

    landmarks = hand_landmarks.landmark

    fingers = 0

    tips = [8, 12, 16, 20]

    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers += 1

    # thumb
    if landmarks[4].x < landmarks[3].x:
        fingers += 1

    return fingers


cap = cv2.VideoCapture(0)

print("Hand Synth Started")
print("Press ESC to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            finger_count = count_fingers(hand_landmarks)

            if finger_count in notes:

                now = time.time()

                if now - last_play > cooldown:
                    play(notes[finger_count])
                    last_play = now

    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture Synth", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()