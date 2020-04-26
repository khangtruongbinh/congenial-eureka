import cv2
import numpy as np
import glob
from PIL import Image
import os
from os.path import exists

### PARAM
fps = 25
in_dir = '/Users/lap01203/Desktop/antimatlab/data/accidents/data_traffic_640px'
out_dir = '/Users/lap01203/Desktop/antimatlab/data/accidents/data_traffic_frames_640'
class_names = ['Normal', 'RoadAccidents']
### PARAM

if not exists("{}".format(out_dir)):
    os.mkdir("{}".format(out_dir))
for class_name in class_names:
    if not exists("{}/{}".format(out_dir, class_name)):
        os.mkdir("{}/{}".format(out_dir, class_name))

video_paths = glob.glob('{}/*/*.mp4'.format(in_dir))
for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    class_name = os.path.basename(os.path.dirname(video_path))
    save_dir = os.path.join(out_dir, class_name, video_name)
    if not exists(save_dir): os.mkdir(save_dir)
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)
    i = 0

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            Image.fromarray(frame).save('{}/{:05d}.jpg'.format(save_dir, i))
            i += 1
            # # Display the resulting frame
            # cv2.imshow('Frame',frame)

            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

# Closes all the frames
cv2.destroyAllWindows()