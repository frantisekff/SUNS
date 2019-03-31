
import pickle
import random
from builtins import print, len
from pickle import UnpicklingError
import time

import cv2

import numpy

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


def loadDataFromAllTypeOfFruit(numFromKind, ifTestData):
    for name in fruit:
        if(ifTestData):
            name += "Test"
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

        randomNum = random.sample(range(count), numFromKind)
        print(randomNum)

        for i in randomNum:
            selectItems.append(loadedList[i])

        print(selectItems)
    if(ifTestData):
        tren = open('testData', 'wb')
    else:
        tren = open('trainData', 'wb')

    numpy.random.shuffle(selectItems)  # zamiesanie dat
    for x in selectItems:
        pickle.dump(x, tren)

    tren.close()

loadDataFromAllTypeOfFruit(30, 1) #30 vzoriek z kazdej triedy

print("Test Data Created")

loadDataFromAllTypeOfFruit(210, 0) #210 vzoriek z kazdej triedy

print("Train Data Created")


