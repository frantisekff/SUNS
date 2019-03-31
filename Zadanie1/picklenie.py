import pickle
from builtins import print
from pickle import UnpicklingError

import cv2
import os
import random
import time
import numpy

from Fruit import Fruit

numpy.set_printoptions(threshold=numpy.nan)

# # # # # # # # # # # # #Nacitanie a zapisanie obrazkov do picklov # # # # # # # # # # # # # # # # # # # #

time_start = time.clock()

listOfDir = os.listdir('C:/Users/frantisek.ff/Downloads/Fruit-Images-Dataset-master/Fruit-Images-Dataset-master/Training')
print(listOfDir)
fruit = [i.split()[0] for i in listOfDir]  # necham len prve slovo
print(fruit)
distinct_fruit = sorted(set(fruit))  # zoradeny a unikatny set ovocia
print(distinct_fruit)
listZeros = [0]*len(distinct_fruit)
listChars = list() * len(distinct_fruit)
numberingOfFruit = numpy.arange(1, len(distinct_fruit)+1)
print(listZeros)
temporarylist = []
mapOfFruit = dict(zip(distinct_fruit, listZeros))
mapOfFruitNumbering = dict(zip(distinct_fruit, numberingOfFruit))
print(mapOfFruitNumbering)


def loadAllImgandDump(rootDir):

    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        onlyDirName = dirName.split('\\')[-1]
        onlyDirName = onlyDirName.split(' ')[0]

        if onlyDirName in mapOfFruit.keys():
            filename = onlyDirName

            if mapOfFruit[onlyDirName] > 0:
                file = open(filename, 'rb')
                for x in range(0, mapOfFruit[onlyDirName]):
                    temporarylist.append(pickle.load(file))
                file.close()

            file = open(filename, 'wb')
            for x in range(0, mapOfFruit[onlyDirName]):
                pickle.dump(temporarylist[x], file, pickle.HIGHEST_PROTOCOL)

            mapOfFruit[onlyDirName] += len(fileList)

            for fname in fileList:
                image = cv2.imread(dirName + '/' + fname)
                image = (image / 255 - 0.5)

                tempFruit = Fruit()
                tempFruit.name = fname
                tempFruit.kind = mapOfFruitNumbering[onlyDirName]
                tempFruit.rgb = image

                pickle.dump(tempFruit, file)
            file.close()
            temporarylist = []

        print('Found subdirectory: %s' % subdirList)
        print(len(fileList))

rootDir = 'C:/Users/frantisek.ff/Downloads/Fruit-Images-Dataset-master/Fruit-Images-Dataset-master/Training'
# loadAllImgandDump(rootDir)

lowest_fruit = min(mapOfFruit.values())
# nameFruit = open("nameFruit", 'w')
# nameFruit.write(str(mapOfFruit.keys()))
# nameFruit.close()
#
print('Najmensi pocet ovocia je %s' % lowest_fruit)
print(mapOfFruit)
print('Pocet ovoci je %s' % len(mapOfFruit))


###########################################################################

############################## Nacitanie z picklov a zobrazenie  #############################################

temporarylist = []

infile = open('Apple','rb')

while 1:
    try:
        temporarylist.append(pickle.load(infile))
    except (EOFError, UnpicklingError):
        break

infile.close()

time_elapsed = (time.clock() - time_start)
print(time_elapsed)

def showImageFromPickle(windowName, data, xWindow, yWindow):
    cv2.namedWindow(windowName)        # Create a named window
    cv2.moveWindow(windowName, xWindow,yWindow)  # Move it to (40,30)
    cv2.imshow(windowName, data + 0.5)

showImageFromPickle("Obr 1 ", temporarylist[526].rgb, 50, 50)
showImageFromPickle("Obr 2 ", temporarylist[256].rgb, 100, 50)
showImageFromPickle("Obr 3 ", temporarylist[56].rgb, 50, 100)
showImageFromPickle("Obr 4 ", temporarylist[806].rgb, 50, 100)

cv2.waitKey(0)






