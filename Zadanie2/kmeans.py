import pickle
from pickle import UnpicklingError

import cv2
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from plotly.utils import numpy
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random

import re

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
listNameFruit = list(mapoffruitnumbering.keys())
num_fruit_fromDataSet = 10
num_clusters = 35

def loadTypeOfFruit(file):
    fruitList = []
    infile = open(file, 'rb')

    for x in range(num_fruit_fromDataSet):
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


X = []
X1 = []
for name in mapoffruitnumbering.keys():
    listFruit = []
    listFruit.append(loadTypeOfFruit(name))
    for i in listFruit[0]:
        X.append(i.rgb.flatten())
        X1.append(i.rgb)


def showImageFromPickle(windowName, data, xWindow, yWindow):
    cv2.namedWindow(windowName)        # Create a named window
    cv2.moveWindow(windowName, xWindow,yWindow)  # Move it to (40,30)
    cv2.imshow(windowName, data + 0.5)


R, y = make_blobs(n_samples=800, n_features=3, centers=4)

# Initializing KMeans
kmeans = KMeans(n_clusters=num_clusters, n_jobs=-1)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
# Ctest = C[0]
#
index = 0
for  i in range(0,num_fruit_fromDataSet*48,num_fruit_fromDataSet):

    name = listNameFruit[index]
    mapoffruitnumbering[name] = labels[i:i+num_fruit_fromDataSet]
    index +=1

img = C.reshape((num_clusters,100,100,3))
# showImageFromPickle("Test", img[0], 100,100)
# showImageFromPickle("Test", img[4], 100,100)
# showImageFromPickle("Test", img[10], 100,100)
# print(mapoffruitnumbering[0])
# print(mapoffruitnumbering[4])
# print(mapoffruitnumbering[10])

# cv2.waitKey(0)




create4Histograms(mapoffruitnumbering['Apple'], mapoffruitnumbering['Grape'], mapoffruitnumbering['Kiwi'], mapoffruitnumbering['Lemon'],'Appfel','Appfel','Appfel','Appfel' )
flipped = {}

for key, value in mapoffruitnumbering.items():
    string =np.array2string(value, precision=2, separator=' ')
    if string not in flipped:
        flipped[string] = [key]
    else:
        flipped[string].append(key)

# mx = max(len(x) for x in mapoffruitnumbering.itervalues())
# [k for k, v in mapoffruitnumbering.iteritems() if len(v)==mx]

# num = max(flipped, key=lambda x: len(flipped[x]))

print("Podobne typy ovocia ( Pri pocte klastrov " + str(num_clusters) + " )" )
for k, v in flipped.items():
    print(k , v)


mx = max(len(x) for x in flipped.values())
similaryFruit = [k for k, v in flipped.items() if len(v)==mx]
for i in similaryFruit:
    num = re.search('[0-9]+', i).group()
    img = C[int(num)].reshape((100, 100, 3))
    showImageFromPickle(num, img, random.randint(1,400), random.randint(1,400))

#cv2.waitKey(0)

print(" ")

# showImageFromPickle("fsgndfio", C[0], 100,100)
# cv2.waitKey(0)
# occurances = np.bincount(labels)
# print(np.argmax(occurances))
fig = plt.figure()
ax = Axes3D(fig)
# # ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2])
colours = ['r', 'r', 'b', 'g', 'g', 'g','r', 'r', 'b', 'g', 'g', 'g','r', 'r', 'b', 'g', 'g', 'g','r', 'r','r', 'r', 'b', 'g', 'g', 'g','r', 'r', 'b', 'g', 'g', 'g','r', 'r', 'b', 'g', 'g', 'g','r', 'r','r', 'r', 'b', 'g', 'g', 'g','r', 'r']
#
#
#
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c="r", s=100)
#
#
#
plt.show()