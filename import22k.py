#!/bin/python3.5.2
# Ostap Voynarovskiy and Andrey Akhmetov
# CGML Final
# Nov 16 2018
# Professor Curro

import os
import sys
import glob
import numpy as np
import stempeg
#import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import wavio
from tqdm import tqdm


if len(sys.argv) < 2:
    print("USAGE: import22k.py checkpoint_name [pos outdir prefix]")
    exit(-1)

# Given to us in Wave-U-Net
BATCH_SIZE = 10
NUM_ITER = 1000  # 1020 #change to more when we get a more robust record parser
EPOCHS = 500  # should be like 2000 but really 20 iterations after it stops improoving the loss
# Tell it what gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf


################ Adding My Stuff Below This ################
def loadWeights(file):
    # figure out how to load weights and push them into the model
    return True


def saveWeights(weights):
    # figure out how to extract weights and push them into a file
    return True


def WaveUNet(features):
    # Im making the model do nothing for now so we can debug the workflow
    # features=tf.expand_dims(features,3)#thought this would fix things... it didn't conv 1d needs a tensor of size 3 not 4 so the extra dim just fucked things
    print("############################")

    # Handle upsampling linearly as defined in model M4
    def UpSample(dataIn):
        dataIn = tf.expand_dims(dataIn, axis=1)
        upsampled = tf.image.resize_bilinear(dataIn, [1, dataIn.get_shape().as_list()[2] * 2])
        return upsampled

    # PARAMETERS OF THE MODEL
    convFilters = 24  # num extra filters per layer
    convStride = 1
    convPadding = 'valid'  # 'valid' means none (switch to valid at some point
    LAYERS = 11  # how deep is  your love (for source separation using U nets)
    down = []  # init array for storing the skip connections
    down_kernel_size = 15  # size of kernel on convolutions headed down
    up_kernel_size = 5  # size of convs going up

    print('shape of features ', features.shape)

    l1 = features
    for i in range(LAYERS):
        # perform 1d Conv using the parameters defined above
        l1 = tf.layers.conv1d(l1, convFilters * (i + 1), down_kernel_size, padding=convPadding,
                              activation=tf.nn.leaky_relu)
        print("post conv 1d \t", l1.shape)  # print the shape for sanity sake
        down.append(l1)  # append the convolved layer to the skip connection list

        # downsample
        l1 = l1[:, ::2, :]
        print("l1d \t\t", l1.shape)

    for i in reversed(range(LAYERS-1)):
        # upsampling
        l1 = UpSample(l1)
        l1 = tf.squeeze(l1, 1)  # upsampling introduces a dimention of rank 1 so we need to remove it
        l1 = l1[:, :-1, :]  # linear upsampling with an odd outputs excludes the last upsampled value

        # 1D Convolution going upwards the U (for reducing filter dimention)
        l1 = tf.layers.conv1d(l1, convFilters * (i + 1), up_kernel_size, padding=convPadding,
                              activation=tf.nn.leaky_relu)
        # print('l1\t\t',l1.shape)

        # CROP AND CONCAT
        offset = int(int(down[i].shape[1] - l1.shape[1]) / 2)  # calculate how much needs to be cropped
        l1 = tf.concat([l1, down[i][:, offset:-offset, :]], 2)  # crop the saved layer to perform a skip
        print('concatenated\t', l1.shape)  # print for sanity

    # Shaping to output dimention to play nicely with the loss
    # output to have 2 channels for stereo
    fin = tf.layers.conv1d(l1, 2, 1, activation=tf.nn.tanh)
    print('final layer \t', fin.shape)
    print("############################")
    return fin



sess = tf.Session()

rootpath = os.getcwd()
trainglob = os.path.join(rootpath, 'test22khz', '*.tfrecord')
train_files = glob.glob(trainglob)

dataset = tf.data.TFRecordDataset([train_files], "ZLIB")
dataset = dataset.batch(BATCH_SIZE, True)
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer)
rslt = iterator.get_next()

tfrecord_features = tf.parse_example(rslt,
                                     features={
                                         'shape': tf.FixedLenFeature([], tf.string),
                                         'rate': tf.FixedLenFeature([], tf.int64),
                                         'mix': tf.FixedLenFeature([], tf.string),
                                         'drums': tf.FixedLenFeature([], tf.string),
                                         'bass': tf.FixedLenFeature([], tf.string),
                                         'accomp': tf.FixedLenFeature([], tf.string),
                                         'vocals': tf.FixedLenFeature([], tf.string),
                                     }, name='features')

shape = [-1, 65523, 2]  # [numsongs, numsamples per song, num channels]

mix = tf.decode_raw(tfrecord_features['mix'], tf.float32)
mix = tf.reshape(mix, shape)

drums = tf.decode_raw(tfrecord_features['drums'], tf.float32)
drums = tf.reshape(drums, shape)

bass = tf.decode_raw(tfrecord_features['bass'], tf.float32)
bass = tf.reshape(bass, shape)

accomp = tf.decode_raw(tfrecord_features['accomp'], tf.float32)
accomp = tf.reshape(accomp, shape)

vocals = tf.decode_raw(tfrecord_features['vocals'], tf.float32)
vocals = tf.reshape(vocals, shape)

features = mix  # tf.placeholder(tf.float32, [None,16384,2])  # Should get batch size by 2 array of labels
labels = drums  # tf.placeholder(tf.float32, [None,16384,2])     # Revisit this idk if it's right

print(sess.run(features))
print(sess.run(labels))

labels_predicted = WaveUNet(features)  # this needs to be moved lower

print('Num_params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

# Match the output shape to the ground truth shape for training
off = int(int(labels.shape[1] - labels_predicted.shape[1]) / 2)  # calculate how much needs to be cropped
labels = labels[:, off:-off, :]  # crop what needs to be cropped in a repeatable manner

loss = tf.losses.mean_squared_error(labels, labels_predicted)

lr = .1
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999).minimize(loss)
# lets hope this works first. The paper gives slightly different parameters which we will update to later


saver = tf.train.Saver();

sess.__enter__()
saver.restore(sess, sys.argv[1])
print("Restored. Call this script with python -i to interact with it.")
if(len(sys.argv)==5):
    print("Dumping an example.")
    mc = mix[:,off:-off,:]
    [lp,gt,mx]=sess.run([labels_predicted, labels, mc])
    bigmix = [lp,gt,mx]
    bigmix[0] = bigmix[0].reshape(286820//2,2)
    bigmix[1] = bigmix[1].reshape(286820//2,2)
    bigmix[2] = bigmix[2].reshape(286820//2,2)

    for i in range(int(sys.argv[2])): 
        print(i)
        [lp,gt,mx]=sess.run([labels_predicted, labels, mc])
        #bigmix[0] = np.concatenate((bigmix[0],lp.reshape(286820//2,2)),0)
        #bigmix[1] = np.concatenate((bigmix[1],gt.reshape(286820//2,2)),0)
        #bigmix[2] = np.concatenate((bigmix[2],mx.reshape(286820//2,2)),0)
    for i in range(30): 
        print(i)
        [lp,gt,mx]=sess.run([labels_predicted, labels, mc])
        bigmix[0] = np.concatenate((bigmix[0],lp.reshape(286820//2,2)),0)
        bigmix[1] = np.concatenate((bigmix[1],gt.reshape(286820//2,2)),0)
        bigmix[2] = np.concatenate((bigmix[2],mx.reshape(286820//2,2)),0)

    wavio.write(os.path.join(sys.argv[3], sys.argv[4]+"lp.wav"), bigmix[0], 22050, sampwidth=3)
    wavio.write(os.path.join(sys.argv[3], sys.argv[4]+"gt.wav"), bigmix[1], 22050, sampwidth=3)
    wavio.write(os.path.join(sys.argv[3], sys.argv[4]+"mx.wav"), bigmix[2], 22050, sampwidth=3)


#################### Model is going above this ####################
### Andrey's is below, its kept as a comment for peace of mind ####

'''
sess = tf.Session()
dataset = tf.data.TFRecordDataset(['train/10000.tfrecord'], "ZLIB")
dataset = dataset.batch(BATCH_SIZE, True)
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer)
rslt = iterator.get_next()

tfrecord_features = tf.parse_example(rslt,
                                            features={
                                                'shape': tf.FixedLenFeature([], tf.string),
                                                'rate': tf.FixedLenFeature([], tf.int64),
                                                'mix': tf.FixedLenFeature([], tf.string),
                                                'drums': tf.FixedLenFeature([], tf.string),
                                                'bass': tf.FixedLenFeature([], tf.string),
                                                'accomp': tf.FixedLenFeature([], tf.string),
                                                'vocals': tf.FixedLenFeature([], tf.string),
                                            }, name='features')

shape = [-1, 16384, 2] #[numsongs, numsamples per song, num channels]

mix = tf.decode_raw(tfrecord_features['mix'], tf.float32)
mix = tf.reshape(mix, shape)

drums = tf.decode_raw(tfrecord_features['drums'], tf.float32)
drums = tf.reshape(drums, shape)

bass = tf.decode_raw(tfrecord_features['bass'], tf.float32)
bass = tf.reshape(bass, shape) 

accomp = tf.decode_raw(tfrecord_features['accomp'], tf.float32)
accomp = tf.reshape(accomp, shape)

vocals = tf.decode_raw(tfrecord_features['vocals'], tf.float32)
vocals = tf.reshape(vocals, shape)
print(sess.run(mix))
sess.run(mix)
print(mix)
'''
### End Andrey's code ###
##########################################################################################



