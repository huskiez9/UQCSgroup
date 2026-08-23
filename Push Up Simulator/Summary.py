import cv2
import numpy as np
from motivation import check_milestones, check_target, draw_if_active as draw_motivation, reset as reset_motivation

WHITE = (255, 255, 255)
GREEN = (0, 255, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

def draw_centered_text(frame, text, y, color=(255, 255, 255), scale=1.5, thickness=2):
    """
    Helper function to perfectly center text horizontally on the screen.
    """
    font = cv2.FONT_HERSHEY_PLAIN

    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    
    # Calculate the x coordinate to center the text
    text_x = (frame.shape[1] - text_size[0]) // 2
    
    # Draw black outline
    cv2.putText(frame, text, (text_x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    # Draw main text
    cv2.putText(frame, text, (text_x, y), font, scale, color, thickness, cv2.LINE_AA)


def summary_screen(frame, reps, final_calories_burned, passed_time_sec):
    height, width = frame.shape[:2]


    #Create blurred frame first
    blurred_frame = cv2.GaussianBlur(frame, (53, 53), 0)
    #Create dark frame second
    dark_overlay = np.full_like(frame, (20, 20, 20)) 
    #Mesh them together
    summary_frame = cv2.addWeighted(blurred_frame, 0.4, dark_overlay, 0.6, 0)
    
    
    minutes = int(passed_time_sec // 60)
    seconds = int(passed_time_sec % 60)
    time_string = f"{minutes:02d}:{seconds:02d}"
    cal_string = f"{final_calories_burned:.1f} kcal"

    # Draw Title
    draw_centered_text(summary_frame, "WORKOUT SUMMARY", int(height * 0.20), WHITE, scale=4, thickness=4)
    
    # Draw Stats (spaced out vertically)
    draw_centered_text(summary_frame, f"Total Reps: {reps}", int(height * 0.45), GREEN, scale=3, thickness=3)
    draw_centered_text(summary_frame, f"Calories Burned: {cal_string}", int(height * 0.60), YELLOW, scale=3, thickness=3)
    draw_centered_text(summary_frame, f"Time Elapsed: {time_string}", int(height * 0.75), ORANGE, scale=3, thickness=3)
    

    draw_centered_text(summary_frame, "Press 'Q' to Exit", int(height * 0.95), (200, 200, 200), scale=1.5, thickness=2)
    # draw_centered_text(summary_frame, "Press 'R' to Restart", int(height * 0.90), (200, 200, 200), scale=1.5, thickness=2)

    # Display Loop
    while True:
        cv2.imshow("Session Summary", summary_frame)
        summary_key = cv2.waitKey(1) & 0xFF
        if summary_key == ord("q"):
            cv2.destroyWindow("Session Summary")
            return "quit"
        elif summary_key == ord("r"):
            cv2.destroyWindow("Session Summary")
            return "restart"
