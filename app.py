from flask import Flask, Response, render_template, request
import cv2
import math
import mediapipe as mp
import pickle
import threading

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Variable to hold the current frame for streaming
current_frame = None
frame_lock = threading.Lock()
detect_pose = False  # Flag to control pose detection

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

# Video streaming generator
def generate_video():
    global current_frame
    video = cv2.VideoCapture(0)

    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if detect_pose:  # Only process if detection is enabled
            results = pose_video.process(rgb_frame)

            if results.pose_landmarks:
                pose_label = classifyPose(results)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, pose_label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Store the current frame in the shared variable
        with frame_lock:
            current_frame = frame

    video.release()

# Route to stream video
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', current_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detect_pose
    detect_pose = True
    return "Detection Started"

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detect_pose
    detect_pose = False
    return "Detection Stopped"

# Start the video streaming thread
threading.Thread(target=generate_video, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)
