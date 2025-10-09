from playsound import playsound
import threading

ALARM_SOUND_PATH = "sounds/alarm.wav"

def play_alarm():
    def _play():
        playsound(ALARM_SOUND_PATH)
    threading.Thread(target=_play, daemon=True).start()
