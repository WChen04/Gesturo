import cv2
import mediapipe as mp
import time
import ctypes
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ========== Media Key Control ==========
def send_media_key(key_code):
    ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
    ctypes.windll.user32.keybd_event(key_code, 0, 2, 0)

def lock_screen():
    ctypes.windll.user32.LockWorkStation()

VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
VK_MEDIA_NEXT = 0xB0
VK_MEDIA_PREV = 0xB1

# ========== Gesture Recognizer Setup ==========
BaseOptions = python.BaseOptions
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizer = vision.GestureRecognizer

model_path = "gesture_recognizer.task"
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE
)
recognizer = GestureRecognizer.create_from_options(options)

# ========== Frame to Image Utility ==========
def create_mediapipe_image(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    return mp_image

# ========== Hand Gesture Tracker ==========
def main():
    global recognizer
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_skip_time = time.time()
    COOLDOWN = 1.5
    hands_together_start_time = None

        
    gesture_emojis = {
        "Thumb_Up": "👍",
        "Thumb_Down": "👎",
        "Victory": "✌️",
        "ILoveYou": "🤟",
        "Pointing_Up": "☝️",
        "Closed_Fist": "✊"
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = create_mediapipe_image(rgb)
        result = recognizer.recognize(mp_image)

        overlay = frame.copy()

        # Detect closed fist held for 5 seconds to lock screen
        gesture_for_lock = "Closed_Fist"
        if result.gestures and result.gestures[0][0].category_name == gesture_for_lock:
            if hands_together_start_time is None:
                hands_together_start_time = time.time()
            else:
                elapsed = time.time() - hands_together_start_time
                countdown = max(0, int(5 - elapsed))
                cv2.rectangle(overlay, (10, 90), (310, 140), (0, 0, 0), -1)
                cv2.putText(overlay, f"Locking in: {countdown}s", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
              

                if elapsed >= 5:
                    print("💥 Initiating Self-Destruct Mode!")
                    for i in reversed(range(1, 6)):
                        frame_copy = frame.copy()
                        cv2.rectangle(frame_copy, (400, 300), (900, 400), (0, 0, 0), -1)
                        cv2.putText(frame_copy, f"Self-destruct in {i}...", (420, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.imshow("Gesturo", frame_copy)
                        if cv2.waitKey(1000) & 0xFF == ord('q'):
                            break
                    lock_screen()       
                    hands_together_start_time = None
        else:
            hands_together_start_time = None

        if result.gestures:
            gesture = result.gestures[0][0].category_name
            emoji = gesture_emojis.get(gesture, "")

            # Modern UI panel top-right with emoji
            cv2.rectangle(overlay, (950, 10), (1270, 90), (30, 30, 30), -1)
            cv2.putText(overlay, f"{emoji} {gesture}", (960, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Large center emoji pop-up
            if emoji:
                cv2.putText(overlay, emoji, (frame.shape[1]//2 - 50, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)

            cooldown_passed = (time.time() - last_skip_time) > COOLDOWN
            if gesture == "Thumb_Up" and cooldown_passed:
                print("🔊 Volume Up")
                send_media_key(VK_VOLUME_UP)
                last_skip_time = time.time()
            elif gesture == "Thumb_Down" and cooldown_passed:
                print("🔉 Volume Down")
                send_media_key(VK_VOLUME_DOWN)
                last_skip_time = time.time()
            elif gesture == "Victory" and cooldown_passed:
                print("👉 Victory (Next)")
                send_media_key(VK_MEDIA_NEXT)
                last_skip_time = time.time()
            elif gesture == "ILoveYou" and cooldown_passed:
                print("👈 ILoveYou (Previous)")
                send_media_key(VK_MEDIA_PREV)
                last_skip_time = time.time()
            elif gesture == "Pointing_Up" and cooldown_passed:
                print("☝️ Pointing Up - Launching Spotify")
                import subprocess
                subprocess.Popen(["start", "spotify:"], shell=True)
                last_skip_time = time.time()
            elif gesture == "Open_Palm" and cooldown_passed:
                print("⏸️ Pause")
                send_media_key(0xB3)  # VK_MEDIA_PLAY_PAUSE acts as pause
                last_skip_time = time.time()

        # Apply overlay and show
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.imshow("Gesturo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[🔁] Reloading recognizer model...")
            recognizer = GestureRecognizer.create_from_options(options)
            print("[✅] Reloaded.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
