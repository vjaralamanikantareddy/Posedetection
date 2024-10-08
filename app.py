import cv2
import math
import pickle
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three landmarks
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Function to classify poses
def classifyPose(landmarks):
    left_shoulder = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_wrist = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_shoulder = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_wrist = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    if (left_shoulder and left_wrist and right_shoulder and right_wrist):
        left_slope = (left_wrist.y - left_shoulder.y) / (left_wrist.x - left_shoulder.x + 1e-10)
        right_slope = (right_wrist.y - right_shoulder.y) / (right_wrist.x - right_shoulder.x + 1e-10)

        if abs(left_slope - right_slope) < 0.2:
            return 'T Pose'

    return 'Unknown Pose'

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the first webcam

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose_video:
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            # Resize frame
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_video.process(rgb_frame)

            if results.pose_landmarks:
                pose_label = classifyPose(results)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, pose_label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Use host='0.0.0.0' for accessibility
