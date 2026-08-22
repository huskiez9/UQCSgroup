import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100 
DURATION = 0.15 
FREQUENCY = 880 

# 1. PRE-COMPUTE the tone once when the program starts.
# This saves CPU and ensures the sound is instantly ready to play.
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
BEEP_TONE = 0.5 * np.sin(2 * np.pi * t * FREQUENCY)

def play_beep():

    try:
        # 2. Play the sound. This returns immediately and plays in the background.
        sd.play(BEEP_TONE, samplerate=SAMPLE_RATE)
        
        # 3. We deliberately OMIT sd.wait() so it doesn't freeze your camera feed.
    except Exception as e:
        # Catch any audio device busy errors so it doesn't crash your tracker
        print(f"Audio playback failed: {e}")