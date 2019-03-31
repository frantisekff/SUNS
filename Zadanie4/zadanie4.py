import pickle
from pickle import UnpicklingError


import numpy as np
import tensorflow as tf


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
#-----------------------------------------------------------------#

trainDataNDArray = np.array(train_dataset)
validDataNDArray = np.array(valid_dataset)
testDataNDArray = np.array(test_dataset)

trainLabelsNDArray = np.array(train_labels)
validLabelsNDArray = np.array(valid_labels)
testLabelsNDArray = np.array(test_labels)

image_size = 100
num_labels = 1024

def reformat(dataset, labels):
  dataset = dataset.astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


trainDataNDArray, trainLabelsNDArray = reformat(trainDataNDArray, trainLabelsNDArray)
validDataNDArray, validLabelsNDArray = reformat(validDataNDArray, validLabelsNDArray)
testDataNDArray, testLabelsNDArray = reformat(testDataNDArray, testLabelsNDArray)

print('Training set', trainDataNDArray.shape, trainLabelsNDArray.shape)
print('Validation set', validDataNDArray.shape, validLabelsNDArray.shape)
print('Test set', testDataNDArray.shape, testLabelsNDArray.shape)

batch_size = 128

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Beta for L2 regularization
beta = 0.000001

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size*3))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validDataNDArray)
    tf_test_dataset = tf.constant(testDataNDArray)

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size*3, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases

    # logits = tf.nn.dropout(logits, 0.95)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    # L2 loss
    # reg = tf.nn.l2_loss(weights)

    # loss = tf.reduce_mean(loss + reg * beta)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 1000

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (trainLabelsNDArray.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = trainDataNDArray[offset:(offset + batch_size), :]
    batch_labels = trainLabelsNDArray[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 200 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), validLabelsNDArray))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), testLabelsNDArray))


print("TEsti")
