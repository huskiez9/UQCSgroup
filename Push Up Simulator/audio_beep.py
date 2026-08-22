import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100 # This is the standard audio quality
DURATION = 0.15 # Duration of the beep in seconds (short beep)
FREQUENCY = 880 # Frequency of the beep in Hz (A5 880 Hz frequency note)

def play_beep():
    """
    Play a beep sound using a separate thread to avoid blocking the main program.
    """
    threading.Thread(target=_play_beep_sound, daemon=True).start()

def _play_beep_sound():
    """
    Generate and play a beep sound.
    """
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * t * FREQUENCY) # Generate a sine wave at the specified frequency
    sd.play(tone, samplerate=SAMPLE_RATE)
    sd.wait()  # Wait until this thread's sound has finished playing (so it doesn't block the main loop)