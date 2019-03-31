import pickle
from collections import OrderedDict
from pickle import UnpicklingError


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def createGraph(xValues, yValues, title, ylabel, xlabel):
    plt.rcdefaults()
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    plt.plot(xValues, yValues, 'bo', xValues, yValues, 'k')
    # plt.xticks(numpy.arange(0, 100, step=5))
    for xy in zip(xValues, yValues):  # <--
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')  # <--
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
    numberingOfFruit = np.arange(1, len(distinct_fruit) + 1)
    mapoffruitnumbering = dict(zip(distinct_fruit, numberingOfFruit))
    return mapoffruitnumbering


loadTestData = []
loadTrainData = []

def loadTrainAndTestData():
    fileTrain = open('trainData','rb')
    for i in range(0,6000):
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

test_labels = []
train_labels = []
test_dataset = []
train_dataset = []
valid_labels = []
valid_dataset = []

for i in loadTrainData[:4800]:
    train_dataset.append(i.rgb.flatten())
    train_labels.append(i.kind)

for i in loadTrainData[4800:]:
    valid_dataset.append(i.rgb.flatten())
    valid_labels.append(i.kind)

for i in loadTestData[:1440]:
    test_dataset.append(i.rgb.flatten())
    test_labels.append(i.kind)

train_dataset = np.array(train_dataset)
valid_dataset = np.array(valid_dataset)
test_dataset = np.array(test_dataset)

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)


num_nodes = 1024
batch_size = 800
beta = 1000
image_size = 100
num_labels = 48

def reformat(dataset, labels):
  dataset = dataset.astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#-----------------------------------------------------------------#



def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])





graphForL2andDropout = tf.Graph()
with graphForL2andDropout.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size*3))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size*3, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relu_layer1 = tf.nn.relu(logits_1)
    # Dropout on hidden layer: RELU layer
    keep_prob = tf.placeholder("float")
    relu_layer_dropout = tf.nn.dropout(relu_layer1, keep_prob)

    logits_2 = tf.matmul(relu_layer_dropout, weights_2) + biases_2
    # Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf_train_labels, logits = logits_2))

    # Loss function with L2 Regularization with beta=0.01
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(loss + beta * regularizers)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(loss)


    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_2)

    # Predictions for validation
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    valid_prediction = tf.nn.softmax(logits_2)

    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    test_prediction = tf.nn.softmax(logits_2)



num_steps = 1001
dropout = [0.9, 0.2, 0.5]
listOfTrainAccuracy = []
test_accuracy = {}

graph = graphForL2andDropout
for i in range(len(dropout)):
    dictTrainAccuracy = {}

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        print("pre dropout: " + format(dropout[i]))
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: dropout[i]}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 200 == 0):
                print("Minibatch loss at step {}: {}".format(step, l))
                print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
                print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
        listOfTrainAccuracy.append(dictTrainAccuracy)
        temp_test = round(accuracy(test_prediction.eval(), test_labels), 2)
        test_accuracy[dropout[i]] = temp_test
        print("Test accuracy: {:.1f}".format(temp_test))
test_accuracy = OrderedDict(sorted(test_accuracy.items(), key=lambda t: t[0]))
createGraph(test_accuracy.keys(), test_accuracy.values(), 'Úspešnosť trénovania pre testovacie dáta (100 krokov)', 'Pravdepodobnosť','Velkosť batchu')

