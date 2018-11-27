import os
import subprocess

'''
    This stage extract the features of frames obtained after clustering frames
'''

OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'

input_folder_frame = './Frames'
output_folder = './FrameFeature'

for frames_folder in os.listdir(input_folder_frame):
    print (frames_folder)
    flag =0
    output_folder_frame = os.path.join(output_folder, frames_folder)
    for frames_ in os.listdir(os.path.join(input_folder_frame,frames_folder)):
        framespath = os.path.join(input_folder_frame,os.path.join(frames_folder,frames_))
        if not os.path.exists(output_folder_frame):
            os.mkdir(output_folder_frame)
        c1 = '"{OPENFACE_LOCATION}" -f "{videos}" -out_dir "' + output_folder_frame + '" -2Dfp -3Dfp -pose -aus -gaze'
        com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , videos= framespath)
        subprocess.call(com1, shell=True, executable="/bin/bash")
