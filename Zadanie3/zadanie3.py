import pickle
from pickle import UnpicklingError


import numpy
from numpy.distutils.system_info import numarray_info, accelerate_info
from sklearn import svm

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score

from sklearn.neural_network import MLPClassifier


def createGraph(xValues, yValues, title, ylabel, xlabel):
    plt.rcdefaults()
    plt.plot(xValues, yValues, 'bo', xValues, yValues, 'k')
    # plt.xticks(numpy.arange(0, 100, step=5))
    plt.grid(True)
    plt.margins(0.05)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def createDictWithNamesOfFruit():
    fruit = ''
    nameFruitFile = open('nameFruit', 'r')
    for line in nameFruitFile:  ## iterates over the lines of the file
        fruit += line
        print(line)
    nameFruitFile.close()

    fruit = fruit.replace("'", "")
    fruit = fruit.replace("dict_keys([", "")
    fruit = fruit.replace("])", "")
    fruit = fruit.replace(" ", "")
    fruit = fruit.split(',')
    distinct_fruit = sorted(fruit)
    numberingOfFruit = numpy.arange(1, len(distinct_fruit) + 1)
    mapoffruitnumbering = dict(zip(distinct_fruit, numberingOfFruit))
    return mapoffruitnumbering


loadTestData = []
loadTrainData = []

def loadTrainAndTestData():
    fileTrain = open('trainData','rb')
    while 1:
        try:
            loadTrainData.append(pickle.load(fileTrain))
        except (EOFError, UnpicklingError):
            break
    print("Data nacitane z trainData")
    fileTrain.close()

    fileValid = open('testData','rb')
    while 1:
        try:
            loadTestData.append(pickle.load(fileValid))
        except (EOFError, UnpicklingError):
            break
    print("Data nacitane z TestData")
    fileValid.close()

loadTrainAndTestData()

testLabels = []
trainLabels = []
testData = []
trainData = []

for i in loadTrainData[:5000]:
    trainData.append(i.rgb.flatten())
    trainLabels.append(i.kind)

for i in loadTestData[:1440]:
    testData.append(i.rgb.flatten())
    testLabels.append(i.kind)




def gridSearchForRBF(trainData, trainLabels):
    Cs = [10, 100, 1000, 10000]
    gammas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    param_grid = {'C': Cs, 'gamma': gammas}

    clf_grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=4)

    clf_grid.fit(trainData, trainLabels)

    print("Best Parameters:\n", clf_grid.best_params_)
    print("Best Estimators:\n", clf_grid.best_estimator_)


def gridSearchForLinear(trainData, trainLabels):
    Cs = [ 0.0001, 0.001, 0.01 ]
    param_grid = {'C': Cs}

    clf_grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=4)

    clf_grid.fit(trainData, trainLabels)

    print("Best Parameters:\n", clf_grid.best_params_)
    print("Best Estimators:\n", clf_grid.best_estimator_)

# gridSearchForLinear(trainData[:700], trainLabels[:700])

num = [50, 100, 200, 1000, 5000]
# num = [50, 100,200]

accuracy = []
def rbfNetTets():
    clfRbf = svm.SVC(C=100, gamma=0.0001, kernel='rbf')
    # num = [50, 100]
    # accuracy = [30, 47, 61, 92, 99]

    for i in num:
        clfRbf.fit(trainData[:i], trainLabels[:i])
        # prediction = clfRbf.predict(testData[:1000])
        curennt_accuracy = round(clfRbf.score(testData, testLabels) * 100, 2)
        print("For " + str(i) + " Accuracy: {}%" + str(curennt_accuracy))
        accuracy.append(curennt_accuracy)



def linearNetTets():
    clfL = svm.SVC(C=0.01, kernel='linear')

    for i in num:
        clfL.fit(trainData[:i], trainLabels[:i])
        # prediction = clfL.predict(testData[:1000])
        # print("For "+ str(i) + " Accuracy: {}%".format(clfL.score(testData, testLabels) * 100))
        # accuracy.append(format(round(clfL.score(testData, testLabels) * 100, 2)*100))
        curennt_accuracy = round(clfL.score(testData, testLabels) * 100, 2)
        print("For " + str(i) + " Accuracy: {}%" + str(curennt_accuracy))
        accuracy.append(curennt_accuracy)


# linearNetTets()
# createGraph(num, accuracy, 'Úspešnosť určenia ovocia sieťou SVM(linear)', 'Pravdepodobnosť',
#                 'Počet vzoriek pri trenovani')

accuracy = []

# Test RBF siete na pocte trenovacich vzoriek [50, 100, 200, 1000, 5000] a vykreslenie grafu
# rbfNetTets()
# createGraph(num, accuracy, 'Úspešnosť určenia ovocia sieťou SVM(rbf)', 'Pravdepodobnosť',
#                 'Počet vzoriek pri trenovani')

numpy.set_printoptions(threshold=numpy.inf)


def mlpNetwork():
    mlp = MLPClassifier(hidden_layer_sizes=(350), max_iter=800, random_state=1)
    # num = [50, 100, 200, 1000, 5000]
    # accuracy = []

    for i in num:
        mlp.fit(trainData[:i],trainLabels[:i])
        predictions = mlp.predict(testData)
        # print(confusion_matrix(testLabels, predictions))
        curennt_accuracy = round(mlp.score(testData, testLabels) * 100, 2)
        print("For " + str(i) + " Accuracy: {}%" + str(curennt_accuracy))
        # print(classification_report(testLabels, predictions))
        accuracy.append(curennt_accuracy)



mlpNetwork()

createGraph(num, accuracy, 'Úspešnosť určenia ovocia sieťou MLP(350)', 'Pravdepodobnosť',
                'Počet vzoriek pri trenovani')

print("Test ")