import cv2
import mediapipe as mp

# ========== Gesture Utilities ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURE_EMOJIS = {
    "✊ Fist": "❌",
    "👍 Thumbs Up": "👍",
    "👎 Thumbs Down": "👎",
    "✌️ Peace": "✌️",
    "☝️ Point": "☝️",
    "👋 Wave": "👋",
    "🤟 Rock": "🤟",
}

# ========== Finger Counting ==========
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(1 if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x else 0)

    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y else 0)

    return fingers

# ========== Gesture Recognition ==========
def recognize_gesture(hand_landmarks):
    fingers = count_fingers(hand_landmarks)
    total = sum(fingers)

    thumb, index, middle, ring, pinky = fingers

    if total == 0:
        return "✊ Fist"
    if thumb and not (index or middle or ring or pinky):
        return "👍 Thumbs Up"
    if not thumb and index and middle and not (ring or pinky):
        return "✌️ Peace"
    if not thumb and index and not (middle or ring or pinky):
        return "☝️ Point"
    if index and pinky and thumb and not (middle or ring):
        return "🤟 Rock"
    if not thumb and not index and not middle and not ring and not pinky:
        return "✊ Fist"

    return None

# ========== Hand Gesture Tracker ==========
def main():
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.65,
                        min_tracking_confidence=0.65) as hands:

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        drawing_points = []
        drawing_enabled = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            multi_landmarks = result.multi_hand_landmarks
            multi_handedness = result.multi_handedness

            if multi_landmarks and multi_handedness:
                for idx, hand_landmarks in enumerate(multi_landmarks):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    gesture = recognize_gesture(hand_landmarks)
                    if gesture:
                        emoji = GESTURE_EMOJIS.get(gesture, "")
                        cv2.putText(frame, f"{gesture} {emoji}", (10, 70 + 40 * idx),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # If gesture is ☝️ Point, enable drawing
                    drawing_enabled = (gesture == "✌️ Peace")

                    # If gesture is ✊ Fist, clear canvas
                    if gesture == "🤟 Rock":
                        drawing_points.clear()

                    # Get index fingertip coordinates (landmark 8)
                    index_finger = hand_landmarks.landmark[8]
                    h, w, _ = frame.shape
                    cx, cy = int(index_finger.x * w), int(index_finger.y * h)

                    if drawing_enabled:
                        drawing_points.append((cx, cy))

            # Draw on frame
            for i in range(1, len(drawing_points)):
                cv2.line(frame, drawing_points[i - 1], drawing_points[i], (0, 0, 255), 4)

            cv2.imshow("Gesturo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
