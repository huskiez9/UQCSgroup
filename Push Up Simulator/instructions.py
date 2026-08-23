import cv2
import os

WINDOW_NAME = "Side Push-Up Tracker"

def show_instructions(image_filename="instructions.png"):
    """
    Displays the instructions image and waits (untimed) for the user
    to press 'c' to continue, or 'q' to quit entirely.
    Returns True to continue, False if the user quit.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, image_filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Could not load '{image_filename}' - skipping instructions screen.")
        return True

    while True:
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            return True
        elif key == ord("q"):
            return False