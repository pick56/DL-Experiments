# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
# _DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.
_NUM_VALIDATION = 3500

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
# 将每个数据集分割的分片数
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path.join(dataset_dir, 'dogsVScats')
  directories = []
  class_names = []
  # 获取该目录下所有文件夹名称
  for filename in os.listdir(flower_root):
    # 拼上文件夹名
    path = os.path.join(flower_root, filename)
    # 如果路径是存在的
    if os.path.isdir(path):
      # 增加此路径，后续读里面的图片
      directories.append(path)
      # 文件夹名作为类名
      class_names.append(filename)

  photo_filenames = []
  # 读每个文件目录
  for directory in directories:
    # 获取每个图片的文件名
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

# 获取文件完整路径名
def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'dogsVScats_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  # 不是训练，就是验证
  assert split_name in ['train', 'validation']
  # 数据集分片、每一片的大小
  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
      # 对于每个分片
      for shard_id in range(_NUM_SHARDS):
        # 新获得一个分片名，由路径、split_name和分片名、分片大小组成
        # 之后相同的的分片名会存储在一起
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        # 一个writer
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          # 开始索引
          start_ndx = shard_id * num_per_shard
          # 结束索引，min为了防止索引超出文件总数
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          # 循环该区间内的图片
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            sys.stdout.flush()
            # 读文件,作为二进制读取
            # Read the filename:
            image_data = tf.gfile.GFile(filenames[i], 'rb').read()
            # 获得图片的高和宽
            height, width = image_reader.read_image_dims(sess, image_data)
            # 先去掉文件名,只返回路径,然后再返回文件夹名字,即'E:/aaa/bbb.txt'的结果为aaa
            class_name = os.path.basename(os.path.dirname(filenames[i]))
            # 将类名换为id
            class_id = class_names_to_ids[class_name]
            # 进行tfrecord文件的转换

            example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
            # bytes(filenames[i], 'utf-8'),bytes(class_name, 'utf-8')

            # example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
            # 每个图片都写tfrecord文件
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()



def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

# 进行数据转换
if __name__ == '__main__':
  dataset_dir = 'E:\DATA'
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')

  # 下载数据,对于猫狗分类,数据已经手动下载好了,每一类7500张图片
  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  # 获取图片文件名和类名
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  # 类名映射到类编号,排序后猫在前狗在后,cat dog
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  # 随机数
  random.seed(_RANDOM_SEED)
  # 洗乱
  random.shuffle(photo_filenames)
  # num往后的都训练，num往前的都验证
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  # 数据转换
  _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)

  # Finally, write the labels file:
  # 生成lable文件id映射类名
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  # 生成lable文件
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # 清除多余文件
  # _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')

