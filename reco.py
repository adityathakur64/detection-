import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        h, w, _ = frame.shape

        landmarks = results.pose_landmarks.landmark

        def label(part, name):
            x, y = int(part.x * w), int(part.y * h)
            cv2.putText(frame, name, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        label(landmarks[mp_pose.PoseLandmark.NOSE], "Nose")
        label(landmarks[mp_pose.PoseLandmark.LEFT_WRIST], "Left Hand")
        label(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], "Right Hand")
        label(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], "Left Leg")
        label(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE], "Right Leg")

    cv2.imshow("Human Body Parts Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






