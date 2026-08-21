"""
STEP 1: Just prove hand tracking works.
Run this first. You should see your webcam feed with 21 dots + skeleton lines
drawn on your hand in real time. This is your "wow it's real" moment.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text
from mediapipe.tasks.python import audio

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# max_num_hands=2 because we'll want two hands later for zoom gestures
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mirror the frame so it feels natural (like looking in a mirror)
    frame = cv2.flip(frame, 1)

    # MediaPipe wants RGB, OpenCV gives us BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Air Whiteboard - Step 1: Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
