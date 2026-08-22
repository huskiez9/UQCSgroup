import threading
from playsound import playsound

BEEP_FILE = "beep.wav"  # Path to the beep sound file

def play_beep():
    threading.Thread(target=__playsound, args=(BEEP_FILE,), daemon=True).start()
