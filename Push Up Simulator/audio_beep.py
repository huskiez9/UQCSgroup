import numpy as np
import sounddevice as sd
import threading

SAMPLE_RATE = 44100
DURATION = 0.15
FREQUENCY = 2000 #880 cycles per second. More cycles higher pitch



t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False) #6615 samples taken in the duration
BEEP_TONE = 0.5 * np.sin(2 * np.pi * FREQUENCY * t) #Creates a sign graph at 6615 samples. Increase frequency to make it high pitch



def play_beep():
    # Play beep on another thread so the camera does not freeze
    threading.Thread(
        target=sd.play,
        args=(BEEP_TONE, SAMPLE_RATE),
        daemon=True
    ).start()