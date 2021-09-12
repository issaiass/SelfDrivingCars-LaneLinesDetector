# Neccesary imports

from SDC.LaneLinesDetector import LaneLinesDetector
import numpy as np
import cv2
import os


# Initialization

#name = 'challenge.mp4'                                        # Video to read
#name = 'solidWhiteRight.mp4'
name = 'solidYellowLeft.mp4'
video_input_path = os.path.join('test_videos', name)           # input video path
video_output_path = os.path.join('test_videos_output', name)   # output video path
cap = cv2.VideoCapture(video_input_path)                       # open the video
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                    # get the widht and height as integers, copy the same...
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                     
fps = int(cap.get(cv2.CAP_PROP_FPS))                           # ... fps to the video writer
print((h,w))

out = cv2.VideoWriter(video_output_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w,h)) # open an output video operation

# get initial parameters
ysize=h
#SolidWhiteRight
#horizon = 310
#vertices = np.array([[85, ysize], [475, horizon], [900, ysize]])
#SolidYellowLeft
horizon = 300
vertices = np.array([[100, ysize], [500, horizon], [900, ysize]])
#General Solution
#horizon = 250
#vertices = np.array([[93, ysize], [487, horizon], [900, ysize]])
#Challenge
#horizon = 430
#vertices = np.array([[240, ysize-60], [600, horizon], [750, horizon], [1050, ysize-60]])

# Solid white right: 5, 50, 150, 4, np.pi/180, 15, 40, 19
# Solid white left: 5, 50, 150, 1, np.pi/180, 15, 60, 30
# Challenge: 5, 50, 255, 1, np.pi/180, 15, 60, 30
key_vals_pair = {'ksize': 5, 'low_thr': 50, 'high_thr': 150,                                                # canny edge detector parameters
                 'rho': 1, 'theta': np.pi/180, 'threshold': 15, 'min_line_lenght': 60, 'max_line_gap': 30,  # hough lines parameters
                 'color':(0, 0, 255), 'thickness': 20}                                                      # drawing paramters
params = dict(key_vals_pair)                                   # get the params
ld = LaneLinesDetector()                                       # load the instance

# Lane Lines Detection
while(cap.isOpened()):                                         # Read until video is completed
    ret, frame = cap.read()                                    # capture frame by frame
    if ret == True:                                            # if there is image returned
        frame = ld.process(frame, params, vertices, avg=True)  # get the processed frame
        cv2.imshow('LaneLinesDetector', frame)                 # show the video
        out.write(frame)                                       # write the current output
        if cv2.waitKey(10) & 0xFF == ord('q'):                 # ESC sequence by 'q' key
            break                                              # Break the loop
    else:                                                      
        break                                                  # Break 
cap.release()                                                  # release the capture instance
out.release()                                                  # release the writting instance 