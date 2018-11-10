# Data import pipeline for audio processing final project
# Imports all files from the given directory, producing TFRecord files for the training and test datasets.
# The specified input directory should contain subdirectories 'train' and 'test'.
# STEMS MP4 files will be read from each of these directories and stored in the corresponding record files.
# Import will occur from current directory, unless another is specified as a commandline argument.

# TFRecord code is adapted from examples in the Stanford Tensorflow Tutorials
# (https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/2017/examples/09_tfrecord_example.py),
# (c) 2017 Huyen Nguyen under MIT License

# from memory_profiler import profile
import os
import sys
import glob
import numpy as np
import stempeg
import tensorflow as tf
import gc
from tqdm import tqdm

rootpath = os.getcwd()
if len(sys.argv) >= 2:
    rootpath = sys.argv[1]

trainglob = os.path.join(rootpath, 'train', '*.stem.mp4')
testglob = os.path.join(rootpath, 'test', '*.stem.mp4')

train_files = glob.glob(trainglob)
test_files = glob.glob(testglob)

FRAGMENT_LENGTH = 16384


def serialize_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Reads a STEMS file, splits it into 16384-sample fragments, and stores these fragments into a TFrecord file.
def process_file(filename, writer):
    # print(filename)
    gc.collect()
    S, rate = stempeg.read_stems(filename, np.float32)

    # print(np.array(S.shape, np.int32)[1:])
    samples = S.shape[1]
    # print("File has %d samples"%samples)
    for i in range(0, samples, FRAGMENT_LENGTH):
        if i + FRAGMENT_LENGTH <= samples:
            # Work around https://github.com/faroit/stempeg/issues/8
            example = write_segment(S[:, i:i + FRAGMENT_LENGTH].astype(np.float32), rate)
            writer.write(example.SerializeToString())


# @profile
def write_segment(S, rate):
    example = tf.train.Example(features=tf.train.Features(feature={
        'shape': serialize_bytes(np.array(S.shape, np.int32)[1:].tobytes()),
        'rate': serialize_int64(rate),
        'mix': serialize_bytes(S[0].tobytes()),
        'drums': serialize_bytes(S[1].tobytes()),
        'bass': serialize_bytes(S[2].tobytes()),
        'accomp': serialize_bytes(S[3].tobytes()),
        'vocals': serialize_bytes(S[4].tobytes()),
    }))
    # print("Wrote record of size "+str(S.shape))
    return example


def process_file_list(file_list, record_path):
    i = 10000
    for filename in tqdm(file_list):
        writer = tf.python_io.TFRecordWriter(os.path.join(record_path, str(i) + '.tfrecord'))
        process_file(filename, writer)
        writer.close()
        i = i + 1


print("Processing train data...")
process_file_list(train_files, 'train')
print("Processing test data...")
process_file_list(test_files, 'test')
