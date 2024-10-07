from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pickle
import time

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Reduce frame rate by adding a delay
FRAME_RATE = 5  # Process one frame every 5 frames
FRAME_COUNT = 0

# Function to classify pose (T-Pose in this case)
def classify_pose(landmarks):
    left_shoulder = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_wrist = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_shoulder = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_wrist = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Calculate slopes to determine if it’s a T-pose
    if left_shoulder and left_wrist and right_shoulder and right_wrist:
        left_slope = (left_wrist.y - left_shoulder.y) / (left_wrist.x - left_shoulder.x + 1e-10)
        right_slope = (right_wrist.y - right_shoulder.y) / (right_wrist.x - right_shoulder.x + 1e-10)

        if abs(left_slope - right_slope) < 0.2:
            return 'T Pose'
    return 'Unknown Pose'

# Video feed generator function
def generate():
    global FRAME_COUNT
    video = cv2.VideoCapture(0)
    
    while True:
        success, frame = video.read()
        if not success:
            break

        FRAME_COUNT += 1
        # Process every FRAME_RATE frames to reduce CPU usage
        if FRAME_COUNT % FRAME_RATE == 0:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose_video.process(frame_rgb)

            if results.pose_landmarks:
                pose_label = classify_pose(results)
                cv2.putText(frame, pose_label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
