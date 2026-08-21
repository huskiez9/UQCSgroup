"""
STEP 1 (Tasks API version): Prove hand tracking works, using the same
API your teammate already set up (mediapipe.tasks), so everyone's code
is compatible.

Requires hand_landmarker.task in the same folder as this script.
Download it from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# --- Set up the hand landmarker ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,  # VIDEO mode = frame-by-frame, good for webcam loops
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Utility for drawing the skeleton, since Tasks API doesn't include
# a built-in draw_landmarks helper like the old solutions API did.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def draw_landmarks_on_frame(frame, detection_result, frame_width, frame_height):
    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections (the skeleton lines)
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_px = (int(start.x * frame_width), int(start.y * frame_height))
            end_px = (int(end.x * frame_width), int(end.y * frame_height))
            cv2.line(frame, start_px, end_px, (0, 255, 0), 2)

        # Draw the 21 landmark dots
        for landmark in hand_landmarks:
            px = (int(landmark.x * frame_width), int(landmark.y * frame_height))
            cv2.circle(frame, px, 4, (0, 0, 255), cv2.FILLED)


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_timestamp_ms = 0  # VIDEO mode requires an increasing timestamp per frame

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the frame in MediaPipe's Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Timestamp must strictly increase — use milliseconds since start
    frame_timestamp_ms += 33  # roughly 30fps; fine even if not exact

    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if detection_result.hand_landmarks:
        draw_landmarks_on_frame(frame, detection_result, frame_width, frame_height)

    cv2.imshow("Air Whiteboard - Tasks API - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
