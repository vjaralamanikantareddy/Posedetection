import cv2
import math
import base64
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three landmarks
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

# Convert base64 image to OpenCV format
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    data = request.get_json()
    image_data = data['image']
    image = base64_to_image(image_data)

    # Pose estimation
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose_video:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_video.process(rgb_frame)

        if results.pose_landmarks:
            pose_label = classifyPose(results)
        else:
            pose_label = 'No Pose Detected'

    return jsonify({'pose': pose_label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
