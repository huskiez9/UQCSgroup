import cv2
import time

def run_countdown(cap, seconds=60):
    """
    Runs a countdown timer and shows a countdown overlay before real tracking starts.
    Return True if the countdown is completed, and False if the user presses 'q' to quit.
    """

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame from the camera.")
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
        elapsed_time = time.time() - start_time
        remaining_time = seconds - int(elapsed_time)

        if remaining_time <= 0:
            break

        height, width = frame.shape[:2]
        text1 = str(remaining_time)
        # Top right for the big number (x = width minus 120 pixels, y = 100 pixels down)
        cv2.putText(frame,
                    text1,
                    (width - 120, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA)
        # Top right for the instruction text (x = width minus 400 pixels, y = 200 pixels down)
        text2 = "Get ready to start!"
        cv2.putText(frame,
                    text2,
                    (width - 300, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    4,
                    cv2.LINE_AA)

        cv2.imshow("Countdown", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False  # User pressed 'q' to quit

    return True  # Countdown completed
