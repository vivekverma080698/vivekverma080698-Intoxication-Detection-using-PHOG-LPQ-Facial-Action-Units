import csv
import numpy as np
import random
import math
from sklearn import svm
from sklearn import preprocessing


PHOGDrunkData = []
PHOGFeatureFileDrunk = 'DrunkPHOG.csv'
with open(PHOGFeatureFileDrunk) as raw:
    csv_reader = csv.reader(raw, delimiter=',')
    for row in csv_reader:
        row = [float(i) for i in row]
        PHOGDrunkData.append(row)

PHOGSoberData = []
PHOGFeatureFileSober = 'SoberPHOG.csv'
data = []
with open(PHOGFeatureFileSober) as raw:
    csv_reader = csv.reader(raw, delimiter=',')
    for row in csv_reader:
        row = [float(i) for i in row]
        PHOGSoberData.append(row)


OPNFDrunkData = []
OPNFFeatureFileDrunk = 'DrunkOPNF.csv'
with open(OPNFFeatureFileDrunk) as raw:
    csv_reader = csv.reader(raw, delimiter=',')
    for row in csv_reader:
        row = [float(i) for i in row]
        OPNFDrunkData.append(row)

OPNFSoberData = []
OPNFFeatureFileSober = 'SoberOPNF.csv'
data = []
with open(OPNFFeatureFileSober) as raw:
    csv_reader = csv.reader(raw, delimiter=',')
    for row in csv_reader:
        row = [float(i) for i in row]
        OPNFSoberData.append(row)

LPQDrunkData = []
LPQFeatureFileDrunk = 'DrunkLPQ.csv'
with open(LPQFeatureFileDrunk) as raw:
    csv_reader = csv.reader(raw, delimiter=',')
    for row in csv_reader:
        row = [float(i) for i in row]
        LPQDrunkData.append(row)

LPQSoberData = []
LPQFeatureFileSober = 'SoberLPQ.csv'
data = []
with open(LPQFeatureFileSober) as raw:
    csv_reader = csv.reader(raw, delimiter=',')
    for row in csv_reader:
        row = [float(i) for i in row]
        LPQSoberData.append(row)

def evaluate_on_test_data(X_test,y_test,model=None):
    predictions = model.predict(X_test)
    correct_classifications = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct_classifications += 1
    accuracy = 100*correct_classifications/len(y_test) #Accuracy as a percentage
    return accuracy


def model_1(LPQSoberData,LPQDrunkData,OPNFSoberData,OPNFDrunkData,PHOGSoberData,PHOGDrunkData):
    Xlpq = list(LPQDrunkData)+list(LPQSoberData)
    Xlpq= preprocessing.scale(Xlpq)
    ones = np.ones(len(LPQDrunkData))
    zeros = np.zeros(len(LPQSoberData))
    Ylpq = np.concatenate((zeros, ones))

    clpq = list(zip(Xlpq, Ylpq))
    
    Xopnf = list(OPNFDrunkData)+list(OPNFSoberData)
    Xopnf= preprocessing.scale(Xopnf)
    ones = np.ones(len(OPNFDrunkData))
    zeros = np.zeros(len(OPNFSoberData))
    Yopnf = np.concatenate((zeros, ones))

    copnf = list(zip(Xopnf, Yopnf))

    Xphog = list(PHOGDrunkData)+list(PHOGDrunkData)
    Xphog= preprocessing.scale(Xphog)
    ones = np.ones(len(OPNFDrunkData))
    zeros = np.zeros(len(OPNFSoberData))
    Yphog = np.concatenate((zeros, ones))

    cphog = list(zip(Xphog, Yphog))

    c = list(zip(clpq,copnf,cphog))
    random.shuffle(c)
    clpq,copnf,cphog= zip(*c)

    Xlpq, ylpq=zip(*clpq)
    Xopnf, yopnf=zip(*copnf)
    Xphog, yphog=zip(*cphog)

    print('reached here')
    # # here we have to think about identity overlap

    fraction = 0.75

    num_of_Train_exp = math.floor(len(Xlpq) * fraction)
    Xlpq_train = Xlpq[0:num_of_Train_exp]
    Xlpq_test = Xlpq[num_of_Train_exp:-1]
    Ylpq_train = ylpq[0:num_of_Train_exp]
    Ylpq_test = ylpq[num_of_Train_exp:-1]

    num_of_Train_exp = math.floor(len(Xopnf) * fraction)
    Xopnf_train = Xopnf[0:num_of_Train_exp]
    Xopnf_test = Xopnf[num_of_Train_exp:-1]
    Yopnf_train = yopnf[0:num_of_Train_exp]
    Yopnf_test = yopnf[num_of_Train_exp:-1]

    num_of_Train_exp = math.floor(len(Xphog) * fraction)
    Xphog_train = Xphog[0:num_of_Train_exp]
    Xphog_test = Xphog[num_of_Train_exp:-1]
    Yphog_train = yphog[0:num_of_Train_exp]
    Yphog_test = yphog[num_of_Train_exp:-1]

    modellpq = svm.SVC(C=1, kernel='rbf', gamma=0.01, cache_size=800)
    modellpq.fit(Xlpq_train, Ylpq_train)
    print('Lpq Model is trained')
    predictionslpq = modellpq.predict(Xlpq_test)

    modelopnf = svm.SVC(C=1, kernel='rbf', gamma=0.01, cache_size=800)
    modelopnf.fit(Xopnf_train, Yopnf_train)
    print('Opnf Model is trained')
    predictionsopnf = modelopnf.predict(Xopnf_test)

    modelphog = svm.SVC(C=1, kernel='rbf', gamma=0.01, cache_size=800)
    modelphog.fit(Xphog_train, Yphog_train)
    print('Phog Model is trained')
    predictionsphog = modelphog.predict(Xphog_test)

    count=0
    for a,b,c,d in zip(predictionslpq,predictionsopnf,predictionsphog,Yphog_test):
        if(a+b+c>=2 and d==1):
            count=count+1
        if(a+b+c<2 and d==0):
            count=count+1
    accuracy=count*100/len(Yphog_test)
    print("Accuracy is "+ str(accuracy))


model_1(LPQSoberData,LPQDrunkData,OPNFSoberData,OPNFDrunkData,PHOGSoberData,PHOGDrunkData)
