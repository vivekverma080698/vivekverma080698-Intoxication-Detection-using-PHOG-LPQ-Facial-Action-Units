import os
import subprocess

'''
This stage extract the mask of faces from the frames using OpenFace Library
'''


def crop_Face():
    OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'
    input_folder_frame = './Frames'
    outputFolder = './CroppedFace/'
    garbagefolder = './OpenFace_output'
    counter = 0
    for file in os.listdir(input_folder_frame):
        if not os.path.exists(outputFolder+file):
            os.mkdir(outputFolder+file)
        for frame in os.listdir(os.path.join(input_folder_frame, file)):
            framespath = os.path.join(input_folder_frame, os.path.join(file, frame))
            # print framespath,file ,frame
            outputFolder1 = os.path.abspath(garbagefolder)
            c1 = '"{OPENFACE_LOCATION}" -f "{images}"  -out_dir "'+ outputFolder1 +'" -simalign -simsize 112'
            com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , images= framespath)
            subprocess.call(com1, shell=True)
            name = frame.replace('jpg','bmp')
            cmd2 = 'find ./OpenFace_output -name \'*.bmp\' -exec mv {} "'+ outputFolder+file+"/"+name+'" \;'
            # print cmd2
            subprocess.call(cmd2, shell=True)
            subprocess.call('rm -r OpenFace_output/*', shell=True)
            counter +=1
    print (counter)

crop_Face()
