# References
# https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
# https://www.analyticsvidhya.com/blog/2022/03/pose-detection-in-image-using-mediapipe-library/
# https://theailearner.com/2018/10/15/extracting-and-saving-video-frames-using-opencv-python/
# Frame rate - https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/

import time
import os,sys,math,re

import numpy as np
import cv2
import mediapipe as mp

import matplotlib.pyplot as plt


# Convert the video to images
def preprocess(video_dir,image_dir):
    cap = cv2.VideoCapture(video_dir) 
    
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(amount_of_frames)
    
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break;
        cv2.imwrite(image_dir + 'excercise'+str(i)+'.jpg',frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()


def distance (X,Y):
  return math.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2)

def get_angle(Hips,Knee,Ankle):
    
    A = distance(Knee,Hips)
    K = distance(Ankle,Hips)
    H = distance(Ankle,Knee)
    
    theta = (A**2+H**2-K**2)/(2*A*H)
    
    res = math.degrees(math.acos(theta))
    return res


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
        plt.close(figure)
    else:
        return original_image, results


def detectPose(i, image_pose, pose, pose_dir, draw=False, display=False):

    original_image = image_pose.copy()
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    resultant = pose.process(image_in_RGB)
    
    image_height, image_width, _ = image_in_RGB.shape
    
    hip, knee, ankle = get_coordinates(resultant,image_width,image_height)
    angle = get_angle(ankle,knee,hip)

    # Below is for drawing
    plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');
    plt.savefig(pose_dir+"output"+str(i)+".jpg")
    draw_pose_images(i,pose_dir,resultant,draw,display,original_image)
    
    return ankle, knee, hip, angle 


# Code to get descending and ascending sequence to determine if knee is being bent 
def get_sequence(A):
    
    total_count_per_rep = 8
    error_rate_between_terms = 5
    excercise_count = 0
    threshold = 5
     
    m1 = 0
    reps = 0
    resting = 0
    
    relevant_seq = []
    output_comments = []

    while True:
    
        # Determine the valid sequence 
        idx = m1+1
        while A[m1]>A[idx] and idx<len(A)-1:
            if A[idx]>=140:
                break
            idx+=1
        m2 = idx
    
        # Split sequence to two based on min value
        min_val = min(A[m1:m2])
        min_index = A[m1:m2].index(min_val)
        
        comments = []

        # Choose sequences greater than 1
        if m2>m1+1:
            comments += ["Person is excercising"]*(m2-m1)
            sequence1 = A[m1:m1+min_index] 
            sequence2 = A[m1+min_index:m2]
    
            # Take the absolute value of the difference to ensure the sequence is decreasing or increasing gradually
            incorrect_terms1 = 0
            for idx in range(1,len(sequence1)-1):
                diff = abs(sequence1[idx-1] - sequence1[idx])
                if diff>error_rate_between_terms:
                    incorrect_terms1+=1
        
            incorrect_terms2 = 0
            for idx in range(1,len(sequence2)-1):
                diff = abs(sequence2[idx-1] - sequence2[idx])
                if diff>error_rate_between_terms:
                    incorrect_terms2+=1
        
            if incorrect_terms1<threshold and incorrect_terms2<threshold:
                
                print("Resting time so far : ", resting)
                #print("Valid sequence : ", A[m1:m2])

                excercise_count += 1
                resting = 0
                relevant_seq.append((m1,m2))
                

            # Calculate Reps
            elif excercise_count == total_count_per_rep: # Assume one rep has 10 knee bends
                print("First index is : {}, Last index is : {}".format(first_index,m2))
                
                reps += 1
                excercise_count = 0 


        elif m2-m1==1:
            comments += ["Person is resting"]
            resting += 1
        
        output_comments += comments
        if m2<len(A)-1:
            m1 = m2
        else:
            break;
        
    print("Relevant sequences are : ",relevant_seq) # Multiply with window size to get all frame indexes
    print("Total reps in the above set is :",reps)
    
    return output_comments


def calculate_reps(total_angles,window):
    # total_frames = 6879, approx 25/ sec, now approx 5/sec
    new_angles = []
    
    for idx in range(0,len(total_angles),window):
        avg_angle = sum(total_angles[idx:idx+window])/window
        new_angles.append(avg_angle)

    seq_comments = get_sequence(new_angles)
    return seq_comments


def pose_estimation(img_dir,pose_dir,n_frames):
    
    images = [int(re.findall(r'\d+',word)[0]) for word in os.listdir(img_dir)]
    images.sort()
    
    stretch_count,count,reps = 0,0,0
    stretch = False
    stretch_time = 10
    frame_time = 0.25
    
    total_angles = []

    for idx, img in enumerate(images[:n_frames]):
        image_path = img_dir+"/"+"excercise"+str(img)+".jpg" 
        output = cv2.imread(image_path)
        try:
            ankle, knee, hip, angle = detectPose(idx,output,pose_image,pose_dir,draw=True,display=True)
        except:
            angle = 180

        total_angles.append(angle)    
    
    return total_angles


def create_video(pose_dir,n_frames):
    
    img_arr = []

    images = [int(re.findall(r'\d+',word)[0]) for word in os.listdir(pose_dir)]
    images.sort()

    for idx, img in enumerate(images[:n_frames]):
        image_path = pose_dir+"output"+str(img)+".jpg"
        img = cv2.imread(image_path)
        height, width, layers = img.shape
        size = (width,height)
        img_arr.append(img)
    
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()


def add_text(input_dir,output_dir,comments,window):
    
    seq_comments = np.repeat(comments,window)
    images = [int(re.findall(r'\d+',word)[0]) for word in os.listdir(input_dir)]
    images.sort()

    for idx, img in enumerate(images[:len(seq_comments)]):
        input_path = input_dir+"/"+"excercise"+str(img)+".jpg"
        output = cv2.imread(input_path)

        new_image = cv2.putText(img = img,text = seq_comments[idx],org = (100, 615),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 3.0,color = (0, 0, 0),thickness = 2)

        output_path = output_dir+"/"+"excercise"+str(img)+".jpg"
        cv2.imwrite(image_path, new_image)


def main():
    video_dir = "data/KneeBendVideo.mp4"
    image_dir = "data/raw_img/"
    pose_dir = "data/pose_img/"
    final_dir = "data/final_img"

    n_frames = 150
    window = 5
    start = time.time()

    #preprocess(video_dir,image_dir)

    global mp_pose, pose_image, mp_drawing
    mp_pose = mp.solutions.pose
    pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7,min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    
    total_angles = pose_estimation(image_dir,pose_dir,n_frames)
    seq_comments = calculate_reps(total_angles,window)
    print("Time taken - {}".format(time.time()-start))

    add_text(pose_dir,final_dir,seq_comments,window)
    create_video(final_dir,n_frames)


if __name__=="__main__":
    main()
