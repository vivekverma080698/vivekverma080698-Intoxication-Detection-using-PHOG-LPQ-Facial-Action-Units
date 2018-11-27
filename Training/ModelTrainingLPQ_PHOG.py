import csv
import numpy as np
import random
import math
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


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



DrunkDataFinal=[]
SoberDataFinal=[]
for lpq,ph in zip(LPQDrunkData,PHOGDrunkData):
    final=lpq+ph
    DrunkDataFinal.append(final)
for lpq,ph in zip(LPQSoberData,PHOGSoberData):
    final=lpq+ph
    SoberDataFinal.append(final)


def evaluate_on_test_data(X_test,y_test,model=None):
    predictions = model.predict(X_test)
    correct_classifications = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct_classifications += 1
    accuracy = 100*correct_classifications/len(y_test) #Accuracy as a percentage
    return accuracy


def model_1(Drunk_Data,Sober_Data):
    # print(len(Drunk_Data),len(Sober_Data))
    X = list(Drunk_Data)+list(Sober_Data)
    # print(len(X))
    ones = np.ones(len(Drunk_Data))
    zeros = np.zeros(len(Sober_Data))
    Y = np.concatenate((zeros, ones))
    c = list(zip(X, Y))
    random.shuffle(c)
    X, y = zip(*c)
    print('Data prepared and splitted..')
    # # here we have to think about identity overlap

    fraction = 0.75

    num_of_Train_exp = int(math.floor(len(X) * fraction))
    print(num_of_Train_exp)
    X_train = X[0:num_of_Train_exp]
    X_test = X[num_of_Train_exp:-1]
    Y_train = y[0:num_of_Train_exp]
    Y_test = y[num_of_Train_exp:-1]

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1] ,'C': [1, 10, 18, 36]}]
    scores = ['precision']
    for score in scores:    
        model1 = GridSearchCV(svm.SVC(), tuned_parameters, cv=3,scoring='%s_macro' % score)
        model1.fit(X_train, Y_train)
        print('done')
        print('Best parameters')
        print(model1.best_params_)
        means = model1.cv_results_['mean_test_score']
        stds = model1.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model1.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = Y_test, model1.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        accuracy = evaluate_on_test_data(X_test,Y_test,model1)
        print("Accuracy is ",accuracy)

model_1(DrunkDataFinal,SoberDataFinal)
