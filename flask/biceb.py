from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl counter variables


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

# Define the function to generate frames
def generate_frames():
    counter = 0 
    stage = None
    counter1 = 0 
    stage1 = None
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

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                angle1 = calculate_angle(shoulder1, elbow1, wrist1)

                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(elbow1, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    print("LEFT="+str(counter))

                if angle1 > 160:
                    stage1 = "down"
                if angle1 < 30 and stage1 =='down':
                    stage1="up"
                    counter1 +=1
                     
                    print("RIGHT="+str(counter1))    

            except:
                pass


            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2) )

            # Add text for counter
            cv2.putText(image, 'LEFT: ' + str(counter), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, 'RIGHT: ' + str(counter1), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Convert image to bytes
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the camera and destroy all windows upon completion
    cap.release()
    cv2.destroyAllWindows()

# Route to render the video stream
@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
