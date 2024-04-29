from flask import Flask, render_template, Response
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
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 



def generate_frames():
    counter = 0 
    stage = None
    cap = cv2.VideoCapture(0)
    target_width = 640
    target_height = 600
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image = cv2.resize(image, (target_width, target_height))  

            # Make detection
            results = pose.process(image)
            results1 = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks (LEFT ARM!!)
         

            # Extract landmarks (RIGHT ARM!)
            try:
                landmarksR = results.pose_landmarks.landmark
                landmarksL = results1.pose_landmarks.landmark

                # Get coordinates
                elbowL= [landmarksL[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarksL[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                shoulderL = [landmarksL[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarksL[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                wristL = [landmarksL[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarksL[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Calculate angle
                angleL = calculate_angle(shoulderL, elbowL, wristL)
                # Get coordinates
                elbowR= [landmarksR[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarksR[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                shoulderR = [landmarksR[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarksR[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                wristR = [landmarksR[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarksR[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculate angle
                angleR = calculate_angle(shoulderR, elbowR, wristR)

                # Visualize angle
                cv2.putText(image, str(angleR),
                            tuple(np.multiply(shoulderR, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(image, str(angleL),
                            tuple(np.multiply(shoulderL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angleR and angleL > 130:
                    stage = "up"
                if angleR and angleL < 20 and stage == 'up':
                      stage = "down"
                      counter += 1
                      print(counter)

            except:
                pass


            cv2.rectangle(image, (0,0), (225,73), (255,255,255), -1)

        # Rep data 
            cv2.putText(image, 'JUMP JACKS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                    (10,70), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
        
        # Stage data  
            cv2.putText(image, ' STAGE', (145,12), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                    (85,70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
        
        
        # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)
            mp_drawing.draw_landmarks(image, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,)

            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
