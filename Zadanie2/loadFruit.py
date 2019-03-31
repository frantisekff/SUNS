############################## Nacitanie z picklov a zobrazenie  #############################################
import pickle
from pickle import UnpicklingError
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import sum, average, sqrt
from collections import Counter


from collections import OrderedDict

# fruitAllDistanceEU1 = []
# fruitAllDistanceEU2 = []
# fruitAllDistanceManhattan = []

def saveData(name_file, data):
    file = open(name_file,'wb')
    pickle.dump(data, file)
    file.close()
    print("Data ulozene do suboru ", name_file)

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

mapoffruitnumbering = createDictWithNamesOfFruit()
mapofFruitAvareages = createDictWithNamesOfFruit()
mapofFruitDiferent = createDictWithNamesOfFruit()

def showImageFromPickle(windowName, data, xWindow, yWindow):
    cv2.namedWindow(windowName)        # Create a named window
    cv2.moveWindow(windowName, xWindow,yWindow)  # Move it to (40,30)
    cv2.imshow(windowName, data + 0.5)


def loadTypeOfFruit(file):
    fruitList = []
    infile = open(file, 'rb')

    for x in range(150):
        try:
            fruitList.append(pickle.load(infile))
        except (EOFError, UnpicklingError):
            print("Error")

    # while 1:
    #     try:
    #         fruitList.append(pickle.load(infile))
    #     except (EOFError, UnpicklingError):
    #         break


    infile.close()
    return fruitList


# test = loadTypeOfFruit('Apple')
# print("TEST")

def findAvarageFruit(fruitList):
    sum2 =  numpy.zeros((100,100,3))

    for i in fruitList:
        for j in fruitList:
            temp1 = i.rgb
            temp2 = j.rgb
            sum = temp1 + temp2
            sum2 = sum2 + sum

    numFruit = len(fruitList)
    print(numFruit)
    avarageOfFruit = sum2 / (numFruit * numFruit)
    return avarageOfFruit

# avarageOfFruit = findAvarageFruit()
# showImageFromPickle("sum2",avarageOfFruit, 100,100)



# for j in fruitList:
#     temp = j.rgb
#     distance = numpy.linalg.norm(avarageOfFruit - temp)
#     fruitAllDistanceEU1.append(distance)
#     # diff = avarageOfFruit - temp  # elementwise for scipy arrays
#     # manhattan = sum(abs(diff))  # Manhattan norm
#     # euklid = sqrt(sum(pow(diff, 2))) # Euclid norm
#     # fruitAllDistanceEU2.append(euklid)
#     # fruitAllDistanceManhattan.append(manhattan)

def manhattanDistance(fruitList):
    allDistances = []
    for i in fruitList:
        for j in fruitList:
            temp1 = i.rgb
            temp2 = j.rgb
        diff = temp1 - temp2  # elementwise for scipy arrays
        manhattan = sum(abs(diff))  # Manhattan norm
        allDistances.append(manhattan)

    return allDistances

def EuclidDistance(fruitList):
    allDistances = []
    for i in fruitList:
        for j in fruitList:
            temp1 = i.rgb
            temp2 = j.rgb

            # distance = numpy.linalg.norm(temp1 - temp2)
            # fruitAllDistanceEU1.append(distance)
            diff = temp1 - temp2  # elementwise for scipy arrays
            euklid = sqrt(sum(pow(diff, 2)))  # Euclid norm
            allDistances.append(euklid)
    print(allDistances)
    return allDistances


def computeAvarages():
    for name in mapoffruitnumbering.keys():
        fruitList = loadTypeOfFruit(name)
        # priemer s kazdym ovocim
        computeWithAvarage = findAvarageFruit(fruitList)
        mapofFruitAvareages[name] = computeWithAvarage

        # prva cast ulohy pre kazdy s kazydym a priemer
        allDistancesForFruit = EuclidDistance(fruitList)
        nameofFile = 'allDistance'+name
        saveData(nameofFile, allDistancesForFruit)
        avarageForType = numpy.average(allDistancesForFruit)
        print(avarageForType)
        mapofFruitDiferent[name] = avarageForType

# computeAvarages()




# saveData('avarageOfFruit', mapofFruitAvareages)
# saveData('diferenceOfFruit', mapofFruitDiferent)


def loadDataAllFruit(name_file):
    infile = open(name_file, 'rb')
    # nacitanie vsetkych picklov so suboru

    mapofFruitAvareages = pickle.load(infile)

    print("Data nacitane z allFruit")
    infile.close()
    return mapofFruitAvareages

# tempr = loadDataAllFruit('avarageOfFruit')
tempr2 = loadDataAllFruit('diferenceOfFruit')


# showImageFromPickle("Apple",tempr['Walnut'], 150, 200)

# less_than_zero = list(filter(lambda x: 10 < x < 20, fruitAllDistance))
# print(less_than_zero)

# spocita kolko je v liste rovnakych hodnot a da ich do dict
# recounted = Counter(fruitAllDistanceEU1)
# max_v = max(recounted.keys())
# # print(max_v)
# print(recounted)

# numbers = list(recounted.keys())
# values = list(recounted.values())

# showImageFromPickle("Obr 1 ", eudistance, 50, 50)




def createBarChart(dictonary, title, ylabel):
    sorted_d = OrderedDict(sorted(dictonary.items(), key=lambda x: x[1]))
    values = list(sorted_d.values())
    keys = list(sorted_d.keys())

    plt.rcdefaults()
    y_pos = np.arange(len(keys))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, keys, rotation='vertical')
    plt.margins(0.005)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()

# createBarChart(tempr2, 'Rozmanitosť druhov ovocí', 'Početnosť')




# # # # # # # # # # # # # # # Histogram 4 x# # # # # # # # # # # # # # # # #

def create4Histograms(data_ax0,data_ax1,data_ax2,data_ax3, title_ax0,title_ax1,title_ax2,title_ax3 ):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()


    plt.title('Histogram vzdialenosti pixelov pre 4 najrozmanitejsie triedy ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    num_bins = 25
    values = list(tempr2.values())
    maxDifernce = max(values)
    # print("Najrozmanitjesie trieda je " + maxDifernce)
    ax0.set_title(title_ax0)
    ax1.set_title(title_ax1)
    ax2.set_title(title_ax2)
    ax3.set_title(title_ax3)
    ax0.set_xlabel('Vzdialenosť bodov')
    ax1.set_xlabel('Vzdialenosť bodov')
    ax2.set_xlabel('Vzdialenosť bodov')
    ax3.set_xlabel('Vzdialenosť bodov')

    ax0.set_ylabel('Početnost')
    ax1.set_ylabel('Početnost')
    ax2.set_ylabel('Početnost')
    ax3.set_ylabel('Početnost')

    ax0.hist(data_ax0, num_bins, facecolor='blue', alpha=0.85)
    ax1.hist(data_ax1, num_bins, facecolor='blue', alpha=0.85)
    ax2.hist(data_ax2, num_bins, facecolor='blue', alpha=0.85)
    ax3.hist(data_ax3, num_bins, facecolor='blue', alpha=0.85)
    fig.tight_layout()
    plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#Vypise prvych n najvacsich cisel
c = Counter(tempr2)
theTop = c.most_common(4)
fruitAllDistacne = loadTypeOfFruit('allDistance' + theTop[0][0])
fruitAllDistacne.append(loadTypeOfFruit('allDistance' + theTop[1][0]))
fruitAllDistacne.append(loadTypeOfFruit('allDistance' + theTop[2][0]))
fruitAllDistacne.append(loadTypeOfFruit('allDistance' + theTop[3][0]))


create4Histograms(fruitAllDistacne[0],fruitAllDistacne[1],fruitAllDistacne[2],fruitAllDistacne[3],theTop[0][0],theTop[1][0],theTop[2][0],theTop[3][0] )

cv2.waitKey(0)