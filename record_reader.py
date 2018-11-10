import os
import sys
import glob
import numpy as np
import stempeg
import tensorflow as tf
import gc
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
sess = tf.Session()

dataset = tf.data.TFRecordDataset(['train/10000.tfrecord'])
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer)
rslt = iterator.get_next()
# k, v = reader.read(tfrecord_file_queue)
# print(k, v)
tfrecord_features = tf.parse_single_example(rslt,
                                            features={
                                                'shape': tf.FixedLenFeature([], tf.string),
                                                'rate': tf.FixedLenFeature([], tf.int64),
                                                'mix': tf.FixedLenFeature([], tf.string),
                                                'drums': tf.FixedLenFeature([], tf.string),
                                                'bass': tf.FixedLenFeature([], tf.string),
                                                'accomp': tf.FixedLenFeature([], tf.string),
                                                'vocals': tf.FixedLenFeature([], tf.string),
                                            }, name='features')
# shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
# print(shape)
# buf = tf.decode_raw(tfrecord_features['mix'], tf.float32)
# buf = tf.reshape(buf, shape)
