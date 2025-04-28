import cv2
import mediapipe as mp
import numpy as np

# ========== Face Mesh Utilities ==========
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ========== Emotion Estimator from Landmarks ==========
def estimate_emotion(face_landmarks, image_width, image_height):
    # Get key landmark indices
    mouth_top = face_landmarks.landmark[13]  # Upper inner lip
    mouth_bottom = face_landmarks.landmark[14]  # Lower inner lip
    mouth_left = face_landmarks.landmark[61]  # Left mouth corner
    mouth_right = face_landmarks.landmark[291]  # Right mouth corner
    left_eyebrow = face_landmarks.landmark[105]
    right_eyebrow = face_landmarks.landmark[334]
    left_eye = face_landmarks.landmark[159]  # Upper eyelid
    right_eye = face_landmarks.landmark[386]  # Upper eyelid

    # Convert to pixel coordinates
    mt_y = int(mouth_top.y * image_height)
    mb_y = int(mouth_bottom.y * image_height)
    ml_x = int(mouth_left.x * image_width)
    mr_x = int(mouth_right.x * image_width)
    le_y = int(left_eye.y * image_height)
    re_y = int(right_eye.y * image_height)
    leb_y = int(left_eyebrow.y * image_height)
    reb_y = int(right_eyebrow.y * image_height)

    # Features
    mouth_open = mb_y - mt_y
    mouth_width = mr_x - ml_x
    left_eyebrow_raise = leb_y - le_y
    right_eyebrow_raise = reb_y - re_y

    # Heuristic rules
    if mouth_open > 25:
        return "Surprised"
    elif mouth_open < 10 and mouth_width > 100:
        return "Happy"
    elif left_eyebrow_raise < -5 and right_eyebrow_raise < -5:
        return "Sad"
    elif left_eyebrow_raise > 10 and right_eyebrow_raise > 10:
        return "Angry"
    else:
        return "Neutral"

# ========== Lightweight Facial Landmark Detection ==========
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_face_mesh.FaceMesh(static_image_mode=False,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

                    h, w, _ = frame.shape
                    emotion = estimate_emotion(face_landmarks, w, h)
                    cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            cv2.imshow("Face Mesh Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()