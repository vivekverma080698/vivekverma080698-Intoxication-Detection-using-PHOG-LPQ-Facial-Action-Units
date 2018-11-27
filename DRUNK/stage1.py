import os
import subprocess

'''
This stage extract features of video using OpenFace library 
videoFeatures takes directories of video folder and output folder 
'''


def videofeatures(input_folder_video, output_folder):
    OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'
    files = []
    for f in os.listdir('./video'):
        files.append(input_folder_video+'/'+f)

    c1 = '"{OPENFACE_LOCATION}" {videos} -out_dir ' + output_folder + ' -2Dfp -3Dfp -pose -aus -gaze'
    videopaths = ""
    for f in range(0, len(files)):
        videopaths += '-f "' + files[f] + '" '

    com1 = c1.format(OPENFACE_LOCATION=OPENFACE_LOCATION, videos=videopaths)
    subprocess.call(com1, shell=True)

# Sample run
input_folder_video = './video'
output_folder = './Features'
videofeatures(input_folder_video,output_folder)
