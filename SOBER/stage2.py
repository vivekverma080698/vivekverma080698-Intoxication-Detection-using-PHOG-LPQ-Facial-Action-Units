import numpy
from sklearn.cluster import KMeans
import cv2
import pandas
import os
import scipy
from sklearn import preprocessing

'''
This stage extract the video frames based on clustering
'''

def processcsv_file():
    input_folder = './Features'
    input_folder_video = './video'
    for f in os.listdir(input_folder):
        filePath = os.path.join(input_folder, f)
        if f[-4:] == '.csv':
            filePathOrig = os.path.join(input_folder_video, f)
            filePathOrig = filePathOrig.replace('.csv', '.mp4')
            do_further_processing_landmark(filePath, filePathOrig)
            # do_further_processing_facs(filePath, filePathOrig)

def do_further_processing_landmark(filePath, filePathOrig):
    try:
        print(filePath)
        with open(filePath) as csvfile:
            reader = pandas.read_csv(csvfile)
            landmarkX = ' eye_lmk_x_'
            landmarkY = ' eye_lmk_y_'
            frame = []
            for index, row in reader.iterrows():
                AU01_c = row[' AU01_c']
                AU02_c = row[' AU02_c']
                AU04_c = row[' AU04_c']
                AU06_c = row[' AU06_c']
                AU07_c = row[' AU07_c']
                AU09_c = row[' AU09_c']
                AU10_c = row[' AU10_c']
                AU12_c = row[' AU12_c']
                AU14_c = row[' AU14_c']
                AU15_c = row[' AU15_c']
                AU17_c = row[' AU17_c']
                AU20_c = row[' AU20_c']
                AU23_c = row[' AU23_c']
                AU25_c = row[' AU25_c']
                AU26_c = row[' AU26_c']
                AU28_c = row[' AU28_c']
                AU45_c = row[' AU45_c']
                if AU01_c or AU02_c or AU04_c or AU06_c or AU07_c or AU09_c or AU10_c or AU12_c or AU14_c or AU15_c or AU17_c or AU20_c or AU23_c or AU25_c or AU26_c or AU28_c or AU45_c:
                    listx = []
                    listy = []
                    for i in range(56):
                        landmarkx = landmarkX + str(i)
                        landmarky = landmarkY + str(i)
                        valuex = row[landmarkx].tolist()
                        listx.append(valuex)
                        valuey = row[landmarky].tolist()
                        listy.append(valuey)
                    listx = listx + listy
                    frame.append(listx)
            # print len(frame[0])
            find_K_cluster(frame, filePathOrig)
    except:
        print('File not Found ',filePath)

# def do_further_processing_facs(filePath, filePathOrig):
#     try:
#         print(filePath)
#         with open(filePath) as csvfile:
#             reader = pandas.read_csv(csvfile)
#             frame = []
#             framenumber = 1
#             for index, row in reader.iterrows():
#                 au_unitList = []
#                 AU01_c = row[' AU01_c']
#                 AU02_c = row[' AU02_c']
#                 AU04_c = row[' AU04_c']
#                 # AU05_c = row[' AU05_c']
#                 # AU06_c = row[' AU06_c']
#                 # AU07_c = row[' AU07_c']
#                 AU09_c = row[' AU09_c']
#                 AU10_c = row[' AU10_c']
#                 AU12_c = row[' AU12_c']
#                 AU14_c = row[' AU14_c']
#                 AU15_c = row[' AU15_c']
#                 AU17_c = row[' AU17_c']
#                 AU20_c = row[' AU20_c']
#                 AU23_c = row[' AU23_c']
#                 AU25_c = row[' AU25_c']
#                 AU26_c = row[' AU26_c']
#                 AU28_c = row[' AU28_c']
#                 AU45_c = row[' AU45_c']
#                 # au_unitList.append(framenumber)
#                 au_unitList.append(AU01_c)
#                 au_unitList.append(AU02_c)
#                 au_unitList.append(AU04_c)
#                 # au_unitList.append(AU05_c)
#                 # au_unitList.append(AU06_c)
#                 # au_unitList.append(AU07_c)
#                 au_unitList.append(AU09_c)
#                 au_unitList.append(AU10_c)
#                 au_unitList.append(AU12_c)
#                 au_unitList.append(AU14_c)
#                 au_unitList.append(AU15_c)
#                 au_unitList.append(AU17_c)
#                 au_unitList.append(AU20_c)
#                 au_unitList.append(AU23_c)
#                 au_unitList.append(AU25_c)
#                 au_unitList.append(AU26_c)
#                 au_unitList.append(AU28_c)
#                 au_unitList.append(AU45_c)
#                 framenumber += 1
#                 frame.append(au_unitList)
#             find_K_cluster(frame, filePathOrig)
#     except:
#         print('File not Found ',filePath)


def find_K_cluster(frame,filePathOrig):
    try:
        print('Processing file ',filePathOrig)
        frame = preprocessing.scale(frame)
        kmeans = KMeans(n_clusters=25, random_state=0).fit(frame)
        frame_number = []
        for cluster in list(kmeans.cluster_centers_):
            values = []
            for framevector in frame:
                values.append(scipy.spatial.distance.euclidean(cluster, framevector))
            frame_number.append(numpy.argmin(values))
        vidcap = cv2.VideoCapture(filePathOrig)
        success,image = vidcap.read()
        count = 0
        framenumberindex=1
        frame_number = [i+1 for i in frame_number]

        direc = './Frames/{}/'.format(filePathOrig[8:-4])
        if not os.path.exists(direc):
            os.mkdir(direc)
        while success:
            if (framenumberindex in frame_number):
                name = './Frames/{}/frame{}.jpg'.format(filePathOrig[8:-4],count)
                cv2.imwrite(name, image)
            success, image = vidcap.read()
            count += 1
            framenumberindex = framenumberindex + 1
    except:
        print("happed something wrong")
        

processcsv_file()
