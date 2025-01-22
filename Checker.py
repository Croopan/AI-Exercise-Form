import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture('Squat.jpg')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
curPosDict = {}

def calcAngle(heel, knee, hip, shoulder):
    hipToShoulder = np.array([shoulder[0] - hip[0], shoulder[2] - hip[2]])
    kneeToHip = np.array([hip[0] - knee[0], hip[2] - knee[2]])
    heelToKnee = np.array([knee[0] - heel[0], knee[2] - heel[2]])
    


    BackBend = np.arccos(np.dot(hipToShoulder, kneeToHip) / (np.linalg.norm(hipToShoulder) * np.linalg.norm(kneeToHip))) 
    KneeTravel = np.arccos(np.dot(heelToKnee, kneeToHip) / (np.linalg.norm(heelToKnee) * np.linalg.norm(kneeToHip)))

    return 180 - BackBend * 180 / np.pi, KneeTravel * 180 / np.pi

with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as pose:
    while cap.isOpened():
        Success, image = cap.read()

        if not Success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_style.get_default_pose_landmarks_style()
        )
        print(result.pose_landmarks)
        if result.pose_landmarks:
            for i, landmark in enumerate(result.pose_landmarks.landmark):
                landmark_name = mp_pose.PoseLandmark(i).name

                curPosDict[landmark_name] = (landmark.x, landmark.y, landmark.z)
            print(calcAngle(curPosDict["RIGHT_HEEL"], curPosDict["RIGHT_KNEE"], curPosDict["RIGHT_HIP"], curPosDict["RIGHT_SHOULDER"]))

        cv2.imshow('MediaPipe Pose Esitimation', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
    
