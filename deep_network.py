# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

# why dropout at RELU
# tune learning rate and momentum, decay
# TODO: check RELU and randomness
# randomize randomness
# play with layers, batches

# stohcastic + 2-layer + L2  = 93.1%
# stohcastic + 2-layer + L2 + dropout = 89.6%

# stohcastic + 3-layer + L2 + with/without dropout = 10%
# stohcastic + 4-layer + L2 + with/without dropout = 10%

""" more layers do not imply a better model. With more layers, you have greater dimensionality.
 Consequently, we are familiar with the term "the curse of dimensionality" where when we increase the model's layers,
we're increasing its dimensionality and hence we need more data. Without more data, your final performance would not be better """

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#################################################################################################
#################################################################################################
# 2 layer Neural Network + L2 reg + dropout + low beta + loads of iteration + learning rate
# ACCURACY: 95.3%
# TODO: momentum, decay. batch size, 4 layer
#############z###################################################################################
#################################################################################################

batch_size = 256
hidden_nodes_1 = 1024
hidden_nodes_2 = 512
hidden_nodes_3 = 256
beta = 0.0006

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # W1
  layer1_weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_nodes_1]))
  # b1
  layer1_biases = tf.Variable(tf.zeros([hidden_nodes_1]))

  # W2
  layer2_weights = tf.Variable(
    tf.truncated_normal([hidden_nodes_1, num_labels]))
  # b2
  layer2_biases = tf.Variable(tf.zeros([num_labels]))

  def add_layerered_network(tf_train_dataset):
      # W1X+b
      input_layer = tf.matmul(tf_train_dataset, layer1_weights) + layer1_biases

      # RELU(WX+b)
      hidden_layer_1 = tf.nn.relu(input_layer)

      # drop'em
      dropped_hidden_layer = tf.nn.dropout(hidden_layer_1, 0.85)

      # W1X+b
      return tf.matmul(dropped_hidden_layer, layer2_weights) + layer2_biases

  def test_network(tf_train_dataset):
    # W1X+b
    input_layer = tf.matmul(tf_train_dataset, layer1_weights) + layer1_biases

    # RELU(WX+b)
    hidden_layer_1 = tf.nn.relu(input_layer)

    return tf.matmul(hidden_layer_1, layer2_weights) + layer2_biases

  logits = add_layerered_network(tf_train_dataset)

  # avg. cross entropy loss
  loss1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # L2 regularization baby!
  loss = tf.reduce_mean(loss1 + beta*(tf.nn.l2_loss(layer1_weights) +
                        tf.nn.l2_loss(layer2_weights)))

  # Decaying learning rate
  global_step = tf.Variable(0)  # count the number of steps taken.
  start_learning_rate = 0.5
  learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    test_network(tf_valid_dataset))
  test_prediction = tf.nn.softmax(test_network(tf_test_dataset))

num_steps = 10000

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
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
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

