import cv2
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture (0 for webcam)
video_capture = cv2.VideoCapture(0)

# Function to classify the T-pose
def classify_pose(landmarks):
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Check if shoulders and wrists are detected
    if (left_shoulder and left_wrist and right_shoulder and right_wrist):
        left_slope = (left_wrist.y - left_shoulder.y) / (left_wrist.x - left_shoulder.x + 1e-10)
        right_slope = (right_wrist.y - right_shoulder.y) / (right_wrist.x - right_shoulder.x + 1e-10)
        if abs(left_slope - right_slope) < 0.2:
            return 'T Pose'
    return 'Unknown Pose'

# Video feed generator function
def gen_frames():
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            # Convert the BGR image to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw the landmarks and pose classification
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                pose_label = classify_pose(results.pose_landmarks)
                cv2.putText(frame, pose_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in HTTP multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
