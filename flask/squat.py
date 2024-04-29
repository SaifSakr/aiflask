from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose




def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
# Define the function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    vec_ab = a - b
    vec_bc = c - b

    dot_product = np.dot(vec_ab, vec_bc)
    magnitude_ab = np.linalg.norm(vec_ab)
    magnitude_bc = np.linalg.norm(vec_bc)

    angle_radians = np.arccos(dot_product / (magnitude_ab * magnitude_bc))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
def generate_frames():
    angle_min = []
    angle_min_hip = []
    counter = 0
    min_ang = 0
    min_ang_hip = 0
    stage = None
    cap = cv2.VideoCapture(0)
    target_width = 640
    target_height = 600
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_ = rescale_frame(frame, percent=100)
            image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image = cv2.resize(image, (target_width, target_height))  

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder1 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                hip1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee1 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle1 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle_knee = calculate_angle(hip, knee, ankle)
                angle_hip = calculate_angle(shoulder, hip, knee)

                angle_knee1 = calculate_angle(hip1, knee1, ankle1)
                angle_hip1 = calculate_angle(shoulder1, hip1, knee1)

                angle_min.append(angle_knee)
                angle_min_hip.append(angle_hip)
                cv2.putText(image, str(angle_knee), 
                           tuple(np.multiply(knee, [1500, 800]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA
                                )
            
                cv2.putText(image, str(angle_hip), 
                           tuple(np.multiply(hip, [1500, 800]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                                )

                if angle_knee > 169 and angle_knee1 > 169:
                    stage = "up"
                if angle_knee <= 90 and angle_knee1 <= 90 and stage == 'up':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                             ) 
                    stage = "down"
                    counter += 1
                    min_ang = min(angle_min)
                    min_ang_hip = min(angle_min_hip)
                    angle_min = []
                    angle_min_hip = []
                    

            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,) 

            
            cv2.putText(image, "Repetition : " + str(counter) + "   stage :" + str(stage),
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Knee-joint angle : " + str(min_ang),
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Hip-joint angle : " + str(min_ang_hip),
                        (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
        


            ret, jpeg = cv2.imencode('.jpg', image)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
