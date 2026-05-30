import mediapipe as mp
import numpy as np 
import pandas as pd 
import cv2

mp_holistic = mp.solutions.holistic
rows_per_frame=543
#face landmarls we get and their associated numbers

NOSE =[1,2,98,327]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0, 61, 185, 40, 39, 37, 267, 269, 270, 409,291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,95, 88, 178, 87, 14, 317, 402, 318, 324, 308,]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,466, 388, 387, 386, 385, 384, 398]
#HANDS 
LHAND= np.arange(468,489).tolist()
RHAND= np.arange(522,543).tolist()
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(POINT_LANDMARKS)#landmarks we care about

CHANNELS = 6*NUM_NODES#we will see as we only consider x,y co ordnitaes but also dx,dx and dx2,dy2
#print(f"number of nodes: {NUM_NODES}")

def landmark_extractor_for_frame(results):
    """
    returns a shape of (543,3) with all landmarks with its [x,y,z] coordinates
    """
    face = np.array(
        [[lm.x,lm.y,lm.z] for lm in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,3))

    left_hand = np.array(
        [[lm.x,lm.y,lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))

    pose = np.array(
        [[lm.x,lm.y,lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))

    right_hand = np.array(
        [[lm.x,lm.y,lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    
    return np.concatenate(
        [face,left_hand,pose,right_hand],
        axis=0
    )

def video_to_parquet(video_path,parquet_path):

    frames_data=[]
     
    NOSE =[1,2,98,327]
    LNOSE = [98]
    RNOSE = [327]
    LIP = [ 0, 61, 185, 40, 39, 37, 267, 269, 270, 409,291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,95, 88, 178, 87, 14, 317, 402, 318, 324, 308,]
    LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
    RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

    POSE = [500, 502, 504, 501, 503, 505, 512, 513]
    LPOSE = [513,505,503,501]
    RPOSE = [512,504,502,500]

    REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,246, 161, 160, 159, 158, 157, 173]
    LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,466, 388, 387, 386, 385, 384, 398]
    #HANDS 
    LHAND= np.arange(468,489).tolist()
    RHAND= np.arange(522,543).tolist()
    POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

    
    point_landmarks = POINT_LANDMARKS
    #open_video 
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5        
    ) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            results = holistic.process(image)
            landmarks = landmark_extractor_for_frame(results)
            landmarks= landmarks[point_landmarks,:]
            frames_data.append(landmarks)

    cap.release()

    frames_data = np.array(frames_data)

    n_frames= frames_data.shape[0]
    n_landmarks = frames_data.shape[1]

    df = pd.DataFrame(frames_data.reshape(n_frames*n_landmarks,3),columns=['x','y','z'])
    df.to_parquet(parquet_path,index=False)
    return frames_data.shape



