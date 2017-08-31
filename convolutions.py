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
num_channels = 1 # grayscale not RGB

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
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

batch_size = 60
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

#    (image-size / (4 * 4) ) * ( image_size * depth )
#   (28-5)/2 == 11
# 28*28/(16) = 49
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  # Covnet Model.
  # A filter/kernel, convolution matrix, or mask is a small matrix
  #
  # (W-F+2P)/S + 1
  # W: input width, F: filter width, P: padding, S: stride
  # S: 1 and 2 (stride2: 1,2,2,1; stride1: 1,1,1,1)
  # P = 0 for valid and P = 1 for same
  #
  # in filter: Length, Width, input channel, output channel
  # in input: batch_size, Length, Width, input channel
  # output: batch_size, output-length, output-width, output channel
  def model(data, keep_prob):
      # CNN1 + MAX-POOL
      conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
      hidden1 = tf.nn.relu(conv + layer1_biases)
      pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      # CNN2 + MAX-POOL
      conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
      hidden2 = tf.nn.relu(conv2 + layer2_biases)
      pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      shape = pool2.get_shape().as_list()
      reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

      # NN1
      hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
      # DROPOUT
      dropout = tf.nn.dropout(hidden3, keep_prob)
      # NN2
      return tf.matmul(dropout, layer4_weights) + layer4_biases

  # Training computation.
  logits = model(tf_train_dataset, 0.6)
  beta = 0.0022 # too much penalty doesn't help
  num_steps = 30001 #95.1 for 50000 steps

  # loss and regularization
  global_step = tf.Variable(0)
  loss1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss = tf.reduce_mean(loss1 + beta*(
        tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)
        + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases)))

  # Learning rate and Optimizer.
  learning_rate = tf.train.exponential_decay(1e-1, global_step, num_steps, 0.7, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))


with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

