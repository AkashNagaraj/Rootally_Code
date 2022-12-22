# References
# https://www.analyticsvidhya.com/blog/2022/03/pose-detection-in-image-using-mediapipe-library/
# https://theailearner.com/2018/10/15/extracting-and-saving-video-frames-using-opencv-python/

import os
import math
import numpy as np

import cv2
import mediapipe as mp

import matplotlib.pyplot as plt


# Convert the video to images
def preprocess(video_dir):
    cap = cv2.VideoCapture(video_dir) 
    
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break;
        cv2.imwrite('data/output/excercise'+str(i)+'.jpg',frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()


def get_angle(hip,knee,ankle):
    
    # cos(C) = (a^2+b^2-c^2)/2ab

    a = np.sum((np.array(knee)-np.array(ankle))**2) 
    b = np.sum((np.array(hip)-np.array(knee))**2) 
    c = np.sum((np.array(hip)-np.array(ankle))**2) 
    C = (a**2+b**2-c**2)/(2*a*b)
    
    print("Vaue of C:",C)
    
    try:
        angle = np.arccos(C)
    except:
        angle = 0
    
    return angle


def get_coordinates(resultant,image_width,image_height):
    
    # We get only left leg values because we assume it is nearest to the camera
    
    left_hip_x_coordinate = resultant.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].x*image_width
    left_hip_y_coordinate = resultant.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].y*image_height

    left_knee_x_coordinate = resultant.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x*image_width
    left_knee_y_coordinate = resultant.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y*image_height

    left_ankle_x_coordinate = resultant.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x*image_width
    left_ankle_y_coordinate = resultant.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y*image_height

    return ((left_hip_x_coordinate,left_hip_y_coordinate),(left_knee_x_coordinate,left_knee_y_coordinate),
            (left_ankle_x_coordinate,left_ankle_y_coordinate))


def draw_pose_images(i, pose_dir, resultant, draw, display, original_image):
    
    if resultant.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),thickness=3, circle_radius=3),connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),thickness=2, circle_radius=2))

    if display:
        plt.figure(figsize=[22,22])
        #plt.subplot(121);plt.imshow(image_pose[:,:,::-1]);plt.title("Input Image");plt.axis('off');
        plt.plot(122);plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');
        plt.savefig(pose_dir+"output"+str(i)+".jpg")
    else:
        return original_image, results



def detectPose(i, image_pose, pose, pose_dir, draw=False, display=False):

    original_image = image_pose.copy()
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    resultant = pose.process(image_in_RGB)
    
    image_height, image_width, _ = image_in_RGB.shape
    hip, knee, ankle = get_coordinates(resultant,image_width,image_height)
    angle = get_angle(hip,knee,ankle)

    print("Calculated angle is : ",angle)

    draw_pose_images(i,pose_dir,resultant,draw,display,original_image)



def pose_estimation(img_dir,pose_dir):
    
    #mp_pose = mp.solutions.pose
    #pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7,min_tracking_confidence=0.7)
    #mp_drawing = mp.solutions.drawing_utils

    images = os.listdir(img_dir)
    for idx, img in enumerate(images):
        image_path = img_dir+"/"+img 
        output = cv2.imread(image_path)
        detectPose(idx,output,pose_image,pose_dir,draw=True,display=True)


def main():
    video_dir = "data/KneeBendVideo.mp4"
    image_dir = "data/raw_img/"
    pose_dir = "data/pose_img/"
    #preprocess(video_dir)
    
    global mp_pose, pose_image, mp_drawing

    mp_pose = mp.solutions.pose
    pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7,min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    pose_estimation(image_dir,pose_dir)


if __name__=="__main__":
    main()
