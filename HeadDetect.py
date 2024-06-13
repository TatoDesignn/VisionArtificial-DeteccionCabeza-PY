import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_pose.Pose(
    static_image_mode=False) as pose:

    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        frame = cv2.flip(frame, 1)
        
        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            x0, y0 = int(landmarks[0].x * width), int(landmarks[0].y * height)  
            x11, y11 = int(landmarks[11].x * width), int(landmarks[11].y * height)  
            x12, y12 = int(landmarks[12].x * width), int(landmarks[12].y * height)  

            
            x_middle = (x11 + x12) // 2
            y_middle = (y11 + y12) // 2

            
            cv2.line(frame, (x_middle, y_middle), (x0, y0), (0, 0, 255), 2)
            
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
            for landmark in [0, 11, 12]:
                x, y = int(landmarks[landmark].x * width), int(landmarks[landmark].y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) 

           
            dx = x0 - x_middle
            dy = y0 - y_middle
            
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            if angle_deg >= -85:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.Canny(gray_frame, threshold1=100, threshold2=200)

            elif angle_deg <= -95:
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()