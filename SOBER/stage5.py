from phogDescriptor import PHogFeatures
import cv2
import pandas
import os
import csv
import numpy
import copy
import sys

'''
This stage extract all the features ex eyeGaze , eye fatigue , phog features
'''

def EyeCheckUp(eye0,eye1):
    # return 50
#     h, w ,ch = eye0.shape
#     originalEye0 = copy.copy(eye0)
#     whitepixel_0=0
#     #
#     image = eye0.reshape((eye0.shape[0] * eye0.shape[1], 3))
#
#     clt = KMeans(n_clusters=3)
#     labels = clt.fit_predict(image)
#     quant = clt.cluster_centers_.astype("uint8")[labels]
#
#     # reshape the feature vectors to images
#     quant = quant.reshape((h, w, 3))
#     image = image.reshape((h, w, 3))
#
#     # convert from L*a*b* to RGB
#     quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
#     image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
#
#     cv2.imshow("Using Kmean", numpy.hstack([eye0, image, quant]))
#
# #<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#     for i in range(0,h):
#         for j in range(0,w):
#             if eye0[i][j][0] > 20 and eye0[i][j][1] > 40 and eye0[i][j][2] > 95 and (max(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2])-min(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2]))>15 and abs(eye0[i][j][2]-eye0[i][j][1])>15 and (eye0[i][j][2]>eye0[i][j][1]) and (eye0[i][j][2]>eye0[i][j][0]):
#                 whitepixel_0 = whitepixel_0 + 1
#                 cv2.circle(eye0, (j, i), 2, (255, 0, 0), -1)
#
#     othercondition = copy.copy(originalEye0)
#     for i in range(0,h):
#         for j in range(0,w):
#             if eye0[i][j][0] > 170 and eye0[i][j][1] > 210 and eye0[i][j][2] > 220 and (max(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2])-min(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2]))>15 and abs(eye0[i][j][2]-eye0[i][j][1])<15 and (eye0[i][j][2]>eye0[i][j][0]) and (eye0[i][j][0]>eye0[i][j][1]):
#                 whitepixel_0 = whitepixel_0 + 1
#                 cv2.circle(othercondition, (j, i), 2, (255, 0, 0), -1)
#
#     cv2.imshow('Using thresholds',numpy.hstack([originalEye0,eye0,othercondition]))
#
# # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#     # eye_cascade = cv2.CascadeClassifier(
#     #     '/home/black/anaconda3/pkgs/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml')
#     #
#     # gray = cv2.cvtColor(eye0, cv2.COLOR_BGR2GRAY)
#     # eyes = eye_cascade.detectMultiScale(gray)
#     # for (ex, ey, ew, eh) in eyes:
#     #     cv2.rectangle(eye0, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
# # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#
    h0, w0 ,ch0 = eye0.shape
    originalEye0 = copy.copy(eye0)
    whitepixel_0=0
    img_yuv0 = cv2.cvtColor(originalEye0, cv2.COLOR_BGR2YUV)
    img_yuv1 = cv2.cvtColor(originalEye0, cv2.COLOR_BGR2YUV)
    img_yuv2 = cv2.cvtColor(originalEye0, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv0[:, :, 0] = cv2.equalizeHist(img_yuv0[:, :, 0])
    img_yuv1[:, :, 1] = cv2.equalizeHist(img_yuv1[:, :, 1])
    img_yuv2[:, :, 2] = cv2.equalizeHist(img_yuv2[:, :, 2])

    # convert the YUV image back to RGB format
    img_output0 = cv2.cvtColor(img_yuv0, cv2.COLOR_YUV2BGR)
    img_output1 = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)
    img_output2 = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)

    method3 = copy.copy(originalEye0)
    for i in range(0,h0):
        for j in range(0,w0):
            if img_output0[i][j][0] > 190 and img_output1[i][j][1] > 100 and img_output2[i][j][2] < 150:
                whitepixel_0 = whitepixel_0 + 1
                # cv2.circle(method3, (j, i), 2, (255, 0, 0), 1)
    #
    # cv2.imshow('YUV ',numpy.hstack([originalEye0,img_output0,img_output1,img_output2,method3]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    h1, w1 ,ch1 = eye1.shape
    originalEye1 = copy.copy(eye1)
    whitepixel_1=0
    img_yuv0 = cv2.cvtColor(originalEye1, cv2.COLOR_BGR2YUV)
    img_yuv1 = cv2.cvtColor(originalEye1, cv2.COLOR_BGR2YUV)
    img_yuv2 = cv2.cvtColor(originalEye1, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv0[:, :, 0] = cv2.equalizeHist(img_yuv0[:, :, 0])
    img_yuv1[:, :, 1] = cv2.equalizeHist(img_yuv1[:, :, 1])
    img_yuv2[:, :, 2] = cv2.equalizeHist(img_yuv2[:, :, 2])

    # convert the YUV image back to RGB format
    img_output0 = cv2.cvtColor(img_yuv0, cv2.COLOR_YUV2BGR)
    img_output1 = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)
    img_output2 = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)

    method3 = copy.copy(originalEye1)
    for i in range(0,h1):
        for j in range(0,w1):
            if img_output0[i][j][0] > 190 and img_output1[i][j][1] > 100 and img_output2[i][j][2] < 150:
                whitepixel_1 = whitepixel_1 + 1
                # cv2.circle(method3, (j, i), 2, (255, 0, 0), 1)

    eye0per = float(whitepixel_0*100)/float((h0*w0))
    eye1per = float(whitepixel_1*100)/float((h1*w1))

    return float((eye0per+eye1per))/2

    # cv2.imshow('YUV ',numpy.hstack([originalEye0,img_output0,img_output1,img_output2,method3]))
    # cv2.waitKey(0)l
    # cv2.destroyAllWindows()


#
# # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#
#     eye1 = cv2.GaussianBlur(originalEye0, (3, 3), 0)
#     eye3 = cv2.cvtColor(eye1, cv2.COLOR_BGR2GRAY)
#     # ret, th1 = cv2.threshold(eye3,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     ret,th1 = cv2.threshold(eye3,100,255,cv2.THRESH_BINARY)
#
#     # cv2.imshow('Binary',scipy.sparse.hstack([originalEye0,th1]))
#     cv2.imshow('Binary',th1)
#     # #
#     # eye2 = cv2.GaussianBlur(eye1, (3, 3), 0)
#     # eye4 = cv2.cvtColor(eye2, cv2.COLOR_BGR2GRAY)
#     # ret,th2 = cv2.threshold(eye4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     #
#     # kernel = numpy.ones((3, 3), numpy.uint8)
#     # th1 = cv2.erode(th1, kernel, iterations=2)
#     # th2 = cv2.erode(th2, kernel, iterations=2)
#     #
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # cimg = cv2.cvtColor(eye0,cv2.COLOR_BGR2GRAY)
    #
    # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,500, param1=50,param2=30,minRadius=0,maxRadius=0)
    #
    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     # draw the outer circle
    #     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #
    # cv2.imshow('detected circles',cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def getGazeFeatures(filePath):
    features = []
    WholeFIle = []
    for cv in os.listdir(filePath):
        if cv[-3:] == 'csv':
            WholeFIle.append(cv)
    WholeFIle.sort()
    for csv in WholeFIle:
        if csv[-3:] == 'csv':
            with open(os.path.join(filePath,csv)) as csvfile:
                reader = pandas.read_csv(csvfile)
                for index, row in reader.iterrows():
                    features.append(row[' gaze_angle_x'])
                    features.append(row[' gaze_angle_y'])
                    features.append(row[' AU01_c'])
                    features.append(row[' AU02_c'])
                    features.append(row[' AU04_c'])
                    features.append(row[' AU06_c'])
                    features.append(row[' AU07_c'])
                    features.append(row[' AU09_c'])
                    features.append(row[' AU10_c'])
                    features.append(row[' AU12_c'])
                    features.append(row[' AU14_c'])
                    features.append(row[' AU15_c'])
                    features.append(row[' AU17_c'])
                    features.append(row[' AU20_c'])
                    features.append(row[' AU23_c'])
                    features.append(row[' AU25_c'])
                    features.append(row[' AU26_c'])
                    features.append(row[' AU28_c'])
                    features.append(row[' AU45_c'])
    return features


def getEyeFatigueValues(filePath):
    feature = []
    counter = 0

    WholeFIle = []
    for csv in os.listdir(filePath):
        if csv[-3:] == 'csv':
            WholeFIle.append(csv)
    WholeFIle.sort()

    for csv in WholeFIle:
        if csv[-3:] == 'csv':
            csvpath = os.path.join(filePath, csv)
            framespath = csvpath.replace('FrameFeature','Frames')
            framespath = framespath.replace('csv', 'jpg')
            with open(csvpath) as csvfile:
                reader = pandas.read_csv(csvfile)
                eye_pupil_X_0 = []
                eye_pupil_y_0 = []
                eye_pupil_X_1 = []
                eye_pupil_y_1 = []

                left_corner_0_x = 0
                right_corner_0_x = 0
                left_corner_1_x = 0
                right_corner_1_x = 0

                for index, row in reader.iterrows():
                    landmark_X_0 = ' eye_lmk_x_'
                    landmark_Y_0 = ' eye_lmk_y_'

                    left_corner_0_x = row[' eye_lmk_x_8']
                    right_corner_0_x = row[' eye_lmk_x_14']
                    left_corner_1_x = row[' eye_lmk_x_36']
                    right_corner_1_x = row[' eye_lmk_x_42']

                    for i in range(8):
                        landmarkx = landmark_X_0 + str(i)
                        landmarky = landmark_Y_0 + str(i)
                        valuex = row[landmarkx].tolist()
                        eye_pupil_X_0.append(valuex)
                        valuey = row[landmarky].tolist()
                        eye_pupil_y_0.append(valuey)
                    landmark_X_1 = ' eye_lmk_x_'
                    landmark_Y_1 = ' eye_lmk_y_'
                    for i in range(28, 36):
                        landmarkx = landmark_X_1 + str(i)
                        landmarky = landmark_Y_1 + str(i)
                        valuex = row[landmarkx].tolist()
                        eye_pupil_X_1.append(valuex)
                        valuey = row[landmarky].tolist()
                        eye_pupil_y_1.append(valuey)

                image = cv2.imread(framespath)
                # for (x, y) in zip(eye_pupil_X_0, eye_pupil_y_0):
                #     cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
                #
                # for (x, y) in zip(eye_pupil_X_1, eye_pupil_y_1):
                #     cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
                # cv2.rectangle(image, (int(left_corner_0_x),int(eye_pupil_y_0[2])), (int(right_corner_0_x), int(eye_pupil_y_0[6])), (0, 255, 0), 1)
                # cv2.rectangle(image, (int(left_corner_1_x),int(eye_pupil_y_1[2])), (int(right_corner_1_x), int(eye_pupil_y_1[6])), (0, 255, 0), 1)
                # cv2.imshow('image',image);
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                h1 = abs(int(eye_pupil_y_0[2]) - int(eye_pupil_y_0[6]))
                w1 = abs(int(right_corner_0_x) - int(left_corner_0_x))
                eye1 = image[int(eye_pupil_y_0[2]):int(eye_pupil_y_0[2]) + h1, int(left_corner_0_x):int(left_corner_0_x) + w1]

                h2 = abs(int(eye_pupil_y_1[2]) - int(eye_pupil_y_1[6]))
                w2 = abs(int(right_corner_1_x) - int(left_corner_1_x))
                eye2 = image[int(eye_pupil_y_1[2]):int(eye_pupil_y_1[2]) + h2, int(left_corner_1_x):int(left_corner_1_x) + w2]
                print (counter,framespath,h1,h2)
                if h1 > 0  and h2 > 0:
                    feature.append(EyeCheckUp(eye1, eye2))
                else:
                    feature.append(0)
                counter += 1
    return feature


def getLPQFeatures(filePath):
    feature = []
    WholeFIle = []
    for csv1 in os.listdir(filePath):
        if csv1[-3:] == 'csv':
            WholeFIle.append(csv1)
    WholeFIle.sort()
    for cv in WholeFIle:
        csvpath = os.path.join(filePath, cv)
        framespath = csvpath.replace('FrameFeature','Frames')
        # print(framespath)
        with open(framespath, 'r') as f:
            reader1 = csv.reader(f)
            fetu_list = list(reader1)
            fetu_list=fetu_list[0]
            fetu_list[-1].replace('\n','')
            # print(len(fetu_list))
            feature = feature + fetu_list
            # print(len(feature))
    return feature

def getPhogFeatures(filePath):
    feature = []
    phog = PHogFeatures()
    WholeFIle = []
    for csv in os.listdir(filePath):
        if csv[-3:] == 'csv':
            WholeFIle.append(csv)

    WholeFIle.sort()
    for csv in WholeFIle:
        csvpath = os.path.join(filePath, csv)
        framespath = csvpath.replace('FrameFeature','Frames')
        framespath = framespath.replace('csv', 'jpg')
        fetu = phog.get_features(framespath)
        feature = feature + list(fetu)
    return feature

def ExtractFeatures():
    input_folder = './FrameFeature'
    phogFeatures_Drunk = []
    lpqFeatures_Drunk = []
    i=0
    f1=open("SoberLPQ.csv","a")
    wr1=csv.writer(f1)
    f2=open("SoberPHOG.csv","a")
    wr2=csv.writer(f2)
    f3=open("SoberOPNF.csv","a")
    wr3=csv.writer(f3)
    for f in os.listdir(input_folder):
        i += 1
        print('Processing...',i)
        filePath = os.path.join(input_folder, f)
        gazeFeature = getGazeFeatures(filePath)
        # eyeFatiguresValues = getEyeFatigueValues(filePath)
        phogFeatures = getPhogFeatures(filePath)
        lpqFeatures = getLPQFeatures(filePath)
        # # print(lpqFeatures)
        wr2.writerow(phogFeatures)
        wr1.writerow(lpqFeatures)
        wr3.writerow(gazeFeature)
        # # phogFeatures_Drunk.append(phogFeatures)
        # lpqFeatures_Drunk.append(lpqFeatures)
        # with open("SoberLPQ.csv", "w") as f:
        #     wr = csv.writer(f)
        #     wr.writerows(lpqFeatures)
        # if i == 2:
        #     break

    # with open("SoberPhog.csv", "w") as f:
    #     wr = csv.writer(f)
    #     wr.writerows(phogFeatures_Drunk)


ExtractFeatures()
