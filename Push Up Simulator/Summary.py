import cv2
import numpy as np
def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.7,
    thickness=2
):
    x, y = position

    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_PLAIN,
        scale, (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_PLAIN,
        scale, color,
        thickness,
        cv2.LINE_AA
    )
WHITE = (255, 255, 255)
GREEN = (0, 255, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

#Summary screen appears when you press e, and when you press q, then it quits summary screen 

def summary_screen (frame, reps, final_calories_burned, passed_time_sec):
    height, width = frame.shape[:2]
    summary_frame = np.full_like(frame, (198,225,245))
    draw_text(summary_frame, "END OF SESSION", (0,0), WHITE, 5, 2)
    draw_text(summary_frame, str(reps), (0, 100), YELLOW, 2, 2)
    draw_text(summary_frame, str(final_calories_burned), (0, 200), YELLOW, 2, 2)
    draw_text(summary_frame, str(passed_time_sec), (0, 300), YELLOW, 2, 2)
    while True:
        cv2.imshow("Session Summary", summary_frame)
        if summary_key == ord("q"):
            break
        summary_key = cv2.waitKey(1) & 0xFF