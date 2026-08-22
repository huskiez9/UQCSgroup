"""
STEP 2: Actually draw in the air.
- Point with JUST your index finger (other fingers curled) = pen down, draws a line
- Open palm (all fingers extended) = eraser, wipes nearby strokes
- Fist (all fingers curled) = pen up, stops drawing (lets you move without a line trailing)

This is intentionally simple geometric logic, no ML classifier needed.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,  # just one hand for this step, add a 2nd later for zoom
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Persistent canvas — this is the "whiteboard" layer. Strokes accumulate here
# frame after frame instead of being wiped each loop.
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

prev_point = None  # last index fingertip position, for drawing continuous lines

# Landmark indices from MediaPipe's hand model (0-20).
# Reference: tip=4/8/12/16/20 for thumb/index/middle/ring/pinky
FINGER_TIPS = [8, 12, 16, 20]      # index, middle, ring, pinky tips
FINGER_PIPS = [6, 10, 14, 18]      # the knuckle joint below each tip


def get_finger_states(landmarks):
    """Returns a list of booleans: is each finger (index, middle, ring, pinky) extended?"""
    states = []
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip_y = landmarks.landmark[tip_idx].y
        pip_y = landmarks.landmark[pip_idx].y
        # In image coordinates, y increases DOWNWARD.
        # A finger is extended if its tip is higher (smaller y) than its knuckle.
        states.append(tip_y < pip_y)
    return states  # e.g. [True, False, False, False] = only index extended


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mode_text = "No hand detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        finger_states = get_finger_states(hand_landmarks)

        # Index fingertip pixel position (this is our "pen tip")
        index_tip = hand_landmarks.landmark[8]
        cx, cy = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

        num_extended = sum(finger_states)

        if finger_states == [True, False, False, False]:
            # ONLY index finger extended -> DRAW mode
            mode_text = "DRAWING"
            if prev_point is not None:
                cv2.line(canvas, prev_point, (cx, cy), (0, 255, 0), 5)
            prev_point = (cx, cy)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        elif num_extended >= 4:
            # All 4 fingers extended -> ERASE mode (open palm)
            mode_text = "ERASING"
            cv2.circle(canvas, (cx, cy), 40, (0, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 2)
            prev_point = None  # lift the pen so we don't draw a line to next point

        else:
            # Fist or any other pose -> pen up, just move
            mode_text = "PEN UP"
            prev_point = None

    else:
        prev_point = None  # hand left the frame, reset so we don't draw a stray line

    # Merge canvas (the drawing) onto the live video feed.
    # Where canvas has drawing (non-black), show it; otherwise show the camera frame.
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    combined = cv2.add(frame_bg, canvas)

    cv2.putText(combined, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.putText(combined, "Press 'c' to clear, 'q' to quit", (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Air Whiteboard - Step 2: Drawing", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
