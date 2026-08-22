import cv2
import time
import threading
import pyttsx3

TRIGGER_REP_COUNT = 67
ANIMATION_DURATION = 2.0 # the seconds the ghost number stays on screen

_triggered_at = None # tracks when the animation started, None = not active

def check_and_trigger(reps):
    """
    Call this once per frame, right after reps is updated
    Starts the animation the instant reps hits exactly 67.
    """
    global _triggered_at
    if reps == TRIGGER_REP_COUNT and _triggered_at is None:
        _triggered_at = time.time() # start the animation
        play_voice()

def draw_if_active (frame):
    """
    Call this every frame wtherer triggered or not
    Draws the ghost number '67' overlay if the animation is currently active
    """
    global _triggered_at
    if _triggered_at is None:
        return

    elapsed = time.time() - _triggered_at
    if elapsed > ANIMATION_DURATION:
        _triggered_at = None # animation finished, and reset for next time
        return

    height, width = frame.shape[:2]
    # Draw the ghost number '67' in the top right corner
    progress = elapsed / ANIMATION_DURATION
    opacity = 1.0 - progress
    scale = 8 + (progress * 2) # starts at 8, grows to 10

    overlay = frame.copy()
    text = '67'
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 15

    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width - text_w) // 2
    y = (height + text_h) // 2

    cv2.putText(overlay, text, (x,y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, dst=frame)

def play_voice():
    threading.Thread(target=_say_six_seven, daemon=True).start()

def _say_six_seven():
    engine = pyttsx3.init()
    engine.say("Six seven!")
    engine.runAndWait()
