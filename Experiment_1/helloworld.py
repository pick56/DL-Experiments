# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 500  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32


def maybe_download(filename):
  """
  Download the data from Yann's website, unless it's already here.
  :param filename:
  :return:
  """
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """
  Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """
  Extract the labels into a vector of int64 label IDs.
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """
  Generate a fake dataset that matches the dimensions of MNIST.
  生成假数据
  """
  data = numpy.ndarray(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """
  Return the error rate based on dense predictions and sparse labels.
  预测和groundtruth之间的误差
  """
  return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


if __name__ == '__main__':
    # 下载数据
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # 6000条训练数据和标签
    train_data = extract_data(train_data_filename, 6000)
    train_labels = extract_labels(train_labels_filename, 6000)
    # 1000条测试数据和标签
    test_data = extract_data(test_data_filename, 1000)
    test_labels = extract_labels(test_labels_filename, 1000)
    # Generate a validation set.
    # 生成验证集
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = 10

    train_size = train_labels.shape[0]
    # 训练集、验证集、测试集
    print(train_data.shape)
    print(train_labels.shape)
    print(validation_data.shape)
    print(validation_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    # 这是训练样本和标签被送到图表的地方
    # 这些占位符节点将在每个节点输入一批训练数据
    # 训练步骤使用{feed_dict}参数进行下面的Run()调用
    # 定义训练数据占位符
    train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # 下面的变量包含所有可训练的权重,他们将会初始化通过下面的方式：
    # {tf.global_variables_initializer().run()}
    # 两个卷积的权重和偏置项
    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))

    # 两个全连接层的权重和偏置项
    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=SEED, dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

    # 定义模型
    def model(data, train=False):
        """模型定义"""
        # 2D 卷积使用same模式，得到的feature map和输入有相同的大小
        # strides是一个4D array格式为[image index, y, x, depth].
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # 最大池化
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Reshape the feature map 变成 2D matrix 以便后面进行全连接
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # 全连接层
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # 训练的时候增加50%的dropout比例。测试和验证的时候不需要进行这一步操作
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # 训练模型定义
    logits = model(train_data_node, True)
    # 定义模型loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits))
    # 定义正则化项，和loss加在一起，这个正则化项只在训练的时候有用
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers

    # 因为训练的时候需要我们不断的减少学习rate
    batch = tf.Variable(0, dtype=data_type())
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # 使用MomentumOptimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

    # 预测当前一个小的batch的结果Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # 预测测试和验证集的结果，Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # 每个batches进行评价，eval_data给验证数据eval_predictions获得预测结果
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        # 截取验证集的一部分，用来获取模型效果
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            # 确定数据足够
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
            # 如果数据不恰好够。倒着取一个batch_size
            else:
                batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # 创建一个会话训练模型
    start_time = time.time()
    with tf.Session() as sess:
        # 随机数初始化变量
        tf.global_variables_initializer().run()
        print('Initialized!')
        # 训练步骤，//表示向下取整除法,10*数据量 // 64
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # 计算数据中当前小批量的偏移量。
            # 请注意，我们可以跨epoche使用更好的随机化。
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            # 这个字典用来传给模型的feed_dict
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            # 使用optimizer更新权重
            sess.run(optimizer, feed_dict=feed_dict)
            # 每100步进行一次验证
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)

