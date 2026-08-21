import cv2
import mediapipe as mp
import numpy as np
import math

# ==========================================================
# WEBCAM
# ==========================================================

cam_width, cam_height = 1280, 720

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ==========================================================
# MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)

mp_draw = mp.solutions.drawing_utils


# ==========================================================
# WINDOW
# ==========================================================

cv2.namedWindow(
    "Drawing",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Drawing",
    cam_width,
    cam_height
)


# ==========================================================
# CANVAS
# ==========================================================

canvas = None


# ==========================================================
# DRAWING SETTINGS
# ==========================================================

brush_thickness = 8

eraser_thickness = 100


# ==========================================================
# SMOOTHING
# ==========================================================

# LOWER = smoother but more delay
# HIGHER = faster but shakier
smoothing_factor = 0.25

smooth_x = None
smooth_y = None


# Previous actual DRAWING position
draw_prev_x = None
draw_prev_y = None


# ==========================================================
# PINCH SETTINGS
# ==========================================================

# Start drawing when below this
PINCH_ON = 55

# Stop drawing only when above this
PINCH_OFF = 75

pinching = False


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape


    # ------------------------------------------------------
    # CREATE CANVAS
    # ------------------------------------------------------

    if canvas is None:

        canvas = np.zeros(
            (h, w, 3),
            dtype=np.uint8
        )


    # ------------------------------------------------------
    # MEDIAPIPE
    # ------------------------------------------------------

    img_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img_rgb.flags.writeable = False

    results = hands.process(img_rgb)

    img_rgb.flags.writeable = True


    # ======================================================
    # HAND DETECTED
    # ======================================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]


        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # --------------------------------------------------
        # LANDMARKS
        # --------------------------------------------------

        thumb_tip = hand_landmarks.landmark[4]

        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]

        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]


        # --------------------------------------------------
        # PIXEL POSITIONS
        # --------------------------------------------------

        tx = int(thumb_tip.x * w)
        ty = int(thumb_tip.y * h)

        raw_x = int(index_tip.x * w)
        raw_y = int(index_tip.y * h)


        # ==================================================
        # SMOOTH FINGER POSITION
        # ==================================================

        if smooth_x is None:

            smooth_x = raw_x
            smooth_y = raw_y

        else:

            smooth_x += (
                raw_x - smooth_x
            ) * smoothing_factor

            smooth_y += (
                raw_y - smooth_y
            ) * smoothing_factor


        curr_x = int(smooth_x)
        curr_y = int(smooth_y)


        # ==================================================
        # PINCH DISTANCE
        # ==================================================

        pinch_distance = math.hypot(
            tx - raw_x,
            ty - raw_y
        )


        # --------------------------------------------------
        # PINCH HYSTERESIS
        # --------------------------------------------------

        if not pinching:

            if pinch_distance < PINCH_ON:

                pinching = True

        else:

            if pinch_distance > PINCH_OFF:

                pinching = False


        # ==================================================
        # FINGER STATES
        # ==================================================

        index_up = (
            index_tip.y < index_pip.y
        )

        middle_up = (
            middle_tip.y < middle_pip.y
        )


        # ==================================================
        # MODES
        # ==================================================

        draw_mode = pinching

        erase_mode = (
            index_up
            and middle_up
            and not pinching
        )

        hover_mode = (
            index_up
            and not middle_up
            and not pinching
        )


        # ==================================================
        # DRAW MODE
        # ==================================================

        if draw_mode:

            if draw_prev_x is None:

                draw_prev_x = curr_x
                draw_prev_y = curr_y


            # ----------------------------------------------
            # INTERPOLATION
            # ----------------------------------------------

            distance_moved = math.hypot(
                curr_x - draw_prev_x,
                curr_y - draw_prev_y
            )


            # Add intermediate points
            steps = max(
                1,
                int(distance_moved / 2)
            )


            old_x = draw_prev_x
            old_y = draw_prev_y


            for i in range(1, steps + 1):

                t = i / steps


                new_x = int(
                    draw_prev_x
                    +
                    (curr_x - draw_prev_x) * t
                )

                new_y = int(
                    draw_prev_y
                    +
                    (curr_y - draw_prev_y) * t
                )


                cv2.line(
                    canvas,
                    (old_x, old_y),
                    (new_x, new_y),
                    (0, 255, 255),
                    brush_thickness,
                    cv2.LINE_AA
                )


                old_x = new_x
                old_y = new_y


            draw_prev_x = curr_x
            draw_prev_y = curr_y


            cv2.circle(
                frame,
                (curr_x, curr_y),
                brush_thickness + 2,
                (0, 255, 255),
                -1,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                "DRAW MODE",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )


        # ==================================================
        # ERASE MODE
        # ==================================================

        elif erase_mode:

            if draw_prev_x is None:

                draw_prev_x = curr_x
                draw_prev_y = curr_y


            cv2.line(
                canvas,
                (draw_prev_x, draw_prev_y),
                (curr_x, curr_y),
                (0, 0, 0),
                eraser_thickness,
                cv2.LINE_AA
            )


            draw_prev_x = curr_x
            draw_prev_y = curr_y


            cv2.circle(
                frame,
                (curr_x, curr_y),
                eraser_thickness // 2,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                "ERASE MODE",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


        # ==================================================
        # HOVER
        # ==================================================

        elif hover_mode:

            cv2.circle(
                frame,
                (curr_x, curr_y),
                brush_thickness + 2,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                "HOVER - Pinch to Draw",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2
            )


            # Stop current stroke
            draw_prev_x = None
            draw_prev_y = None


        else:

            draw_prev_x = None
            draw_prev_y = None


    # ======================================================
    # NO HAND
    # ======================================================

    else:

        draw_prev_x = None
        draw_prev_y = None

        smooth_x = None
        smooth_y = None

        pinching = False


    # ======================================================
    # COMBINE CAMERA + CANVAS
    # ======================================================

    frame = cv2.add(
        frame,
        canvas
    )


    cv2.imshow(
        "Drawing",
        frame
    )


    # ======================================================
    # KEYBOARD
    # ======================================================

    key = cv2.waitKey(1) & 0xFF


    # ESC
    if key == 27:
        break


    # C = clear
    elif key == ord("c"):

        canvas[:] = 0


cap.release()

hands.close()

cv2.destroyAllWindows()