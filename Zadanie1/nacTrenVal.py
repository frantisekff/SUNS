import pickle
import random
from builtins import print, len
from pickle import UnpicklingError
import time

import cv2
import numpy
import numpy as np

from Fruit import Fruit


time_start = time.clock()

loadedList = []
selectItems = []
fruit = ''
nameFruitFile =  open('nameFruit','r')
for line in nameFruitFile:   ## iterates over the lines of the file
    fruit += line
    print (line)

nameFruitFile.close()
fruit = fruit.replace("'", "")
fruit = fruit.replace("dict_keys([", "")
fruit = fruit.replace("])", "")
fruit = fruit.replace(" ", "")


fruit = fruit.split(',')
distinct_fruit = sorted(fruit)
numberingOfFruit = numpy.arange(1, len(distinct_fruit)+1)
mapOfFruitNumbering = dict(zip(numberingOfFruit, distinct_fruit))

############################## Nacitanie y picklov a vytvorenie validacneho a trenovacieho datasetu #############################################

def loadDataFromAllTypeOfFruit():
    for name in fruit:
        print(name)
        count = 0

        infile = open(name, 'rb')
        loadedList = []
        while 1:
            try:
                loadedList.append(pickle.load(infile))
                count += 1
            except (EOFError, UnpicklingError):
                break

        infile.close()

        randomNum = random.sample(range(count), 440)
        print(randomNum)

        for i in randomNum:
            selectItems.append(loadedList[i])

        print(selectItems)

    tren = open('allFruit', 'wb')
    for x in selectItems:
        pickle.dump(x, tren)

    tren.close()

# loadDataFromAllTypeOfFruit()


#  #  #  #  #  #  #  #  #  #  #  # Rozdelenie dat (na train a valid) a test rozdelenia #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

temporarylist = []
loadValidData = []
loadTrainData = []

def loadDataAllFruit():
    infile = open('allFruit', 'rb')
    # nacitanie vsetkych picklov so suboru
    while 1:
        try:
            temporarylist.append(pickle.load(infile))
        except (EOFError, UnpicklingError):
            break
    print("Data nacitane z allFruit")
    infile.close()

def showImageFromPickle(windowName, data, xWindow, yWindow):
    cv2.namedWindow(windowName)        # Create a named window
    cv2.moveWindow(windowName, xWindow,yWindow)  # Move it to (40,30)
    cv2.imshow(windowName, data + 0.5)

# loadDataAllFruit()
# numpy.random.shuffle(temporarylist) # zamiesanie dat
# training, valid = temporarylist[:16800], temporarylist[16800:]


## funkcia ktora rozdeli cely dataset na validacne a trenovacie
def showDivisionofFruit(data, description):
    divisionOfData =  {}
    for i in data:

        if i.kind in divisionOfData.keys():
            pocet = divisionOfData[i.kind]
            pocet +=1
            divisionOfData[i.kind] = pocet
        else:
            divisionOfData.update({i.kind: 1})
    print('Rozdelenie ovocia. ', description, ' data',  divisionOfData)


# showDivisionofFruit(training, 'Trenovacie ')
# showDivisionofFruit(valid, 'Validacne ')


#  #  #  #  #  #  #  #  #  #  #  # Vytvorenie pickle suborov pre train a valid  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
def saveData(name_file, data):
    file = open(name_file,'wb')
    for i in data:
        pickle.dump(i, file)
    file.close()
    print("Data ulozene do suboru ", name_file)

# saveData('validData', valid)
# saveData('trainData', training)

# print(str(temporarylist[214].kind) + " " + str(temporarylist[214].name))
# print(mapOfFruitNumbering[temporarylist[214].kind])
# print(str(temporarylist[1222].kind), str(temporarylist[1222].name))
# print(mapOfFruitNumbering[temporarylist[1222].kind])


# showImageFromPickle(str(temporarylist[214].kind) + " " + str(temporarylist[214].name), temporarylist[214].rgb, 100, 100)
# showImageFromPickle(str(temporarylist[1222].kind) + " " + str(temporarylist[1222].name), temporarylist[1222].rgb, 300, 100)


def loadTrainAndValid():
    fileTrain = open('trainData','rb')
    while 1:
        try:
            loadTrainData.append(pickle.load(fileTrain))
        except (EOFError, UnpicklingError):
            break
    print("Data nacitane z trainData")
    fileTrain.close()

    fileValid = open('validData','rb')
    while 1:
        try:
            loadValidData.append(pickle.load(fileValid))
        except (EOFError, UnpicklingError):
            break
    print("Data nacitane z validData")
    fileValid.close()

loadTrainAndValid()

# print(str(loadTrainData[1002].kind) + " " + str(loadTrainData[1002].name))
# print(mapOfFruitNumbering[loadTrainData[1002].kind])
print(str(loadValidData[5].kind), str(loadValidData[5].name))
print(mapOfFruitNumbering[loadValidData[5].kind])

# showImageFromPickle(str(loadTrainData[1002].kind) + " " + str(loadTrainData[1002].name), loadTrainData[1002].rgb, 100, 100)
showImageFromPickle(str(loadValidData[5].kind) + " " + str(loadValidData[5].name), loadValidData[5].rgb, 400, 100)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)
cv2.waitKey(0)

