#!/bin/python3.5.2
# Ostap Voynarovskiy and Andrey Akhmetov
# CGML Final
# Nov 16 2018
# Professor Curro

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
import glob
import numpy as np
import stempeg
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from tensorflow.contrib.signal import stft, hamming_window
from tqdm import tqdm

# Given to us in Wave-U-Net
BATCH_SIZE = 8
NUM_ITER = 1000  # 1020 #change to more when we get a more robust record parser
EPOCHS = 2000  # should be like 2000 but really 20 iterations after it stops improoving the loss
# Tell it what gpu to use

# Accepts tensors of shape [BatchSize, Samples, Channels].
#  To spectral-loss multiple predictions (e.g. drums and vocals), call this separately on each track.
# NORMALIZED
def phasor_loss(expected, actual):
    eps = 0.0000001
    pwr = 0.5 # hyperparameter on 0..1. 1 = phase ONLY; 0 = unnormalized phasors
    exp = tf.transpose(expected, [0, 2, 1])  # Place the samples in the last dimension as required by stft
    act = tf.transpose(actual, [0, 2, 1])

    estft = stft(exp, 4096, 2048, window_fn=hamming_window, pad_end=True)
    astft = stft(act, 4096, 2048, window_fn=hamming_window, pad_end=True)
    esn = tf.abs(estft) + eps
    asn = tf.abs(astft) + eps
    esph = tf.divide(estft, tf.pow(esn, pwr))
    asph = tf.divide(estft, tf.pow(esn, pwr))
    mag_err = tf.reduce_mean(tf.square(tf.abs(estft-astft)));
    return mag_err


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
trainglob = os.path.join(rootpath, 'train22khz', '*.tfrecord')
train_files = glob.glob(trainglob)
valglob = os.path.join(rootpath, 'train22khz', '*.tfrecord')
validation_files = glob.glob(valglob)

train_dataset = tf.data.TFRecordDataset([train_files], "ZLIB")
train_dataset = train_dataset.batch(BATCH_SIZE, True)
train_iterator = train_dataset.make_initializable_iterator()

val_dataset = tf.data.TFRecordDataset([validation_files], "ZLIB").batch(BATCH_SIZE, True)
val_iterator = val_dataset.make_initializable_iterator()

sess.run(train_iterator.initializer)
sess.run(val_iterator.initializer)

handle = tf.placeholder(tf.string, shape=[])

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

joint_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

rslt = joint_iterator.get_next()

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

#print(sess.run(features))
#print(sess.run(labels))

labels_predicted = WaveUNet(features)  # this needs to be moved lower

print('Num_params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

# Match the output shape to the ground truth shape for training
off = int(int(labels.shape[1] - labels_predicted.shape[1]) / 2)  # calculate how much needs to be cropped
labels = labels[:, off:-off, :]  # crop what needs to be cropped in a repeatable manner

loss = phasor_loss(labels, labels_predicted)

lr = .1
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999).minimize(loss)
# lets hope this works first. The paper gives slightly different parameters which we will update to later

tf.summary.scalar("122L", loss);
init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter("trainlog_22k_Phasey", sess.graph);
saver = tf.train.Saver()
merged = tf.summary.merge_all();
for epo in range(EPOCHS):
    print('EPOCH', epo)
    k = 0
    #for k in tqdm(range(0, NUM_ITER)):
    while True:
        k = k+1
        # x_np, labels_np = data.get_batch() # no more data.getBatch we use the tf records now
        try:
            [loss_np, _, summ] = sess.run([loss, optim, merged],
                                    feed_dict={handle: train_handle})  # , {features:mix,labels:vocals}
            #print(es_);
            #rint(as_);
            if k % 5 == 0:
                print("Loss: " + str(loss_np), file=sys.stderr)
            writer.add_summary(summ, k + epo * NUM_ITER);
        except tf.errors.OutOfRangeError:
            break
    modelName = os.getcwd() + '/models/unet22khzPhasey/' + str(epo) + 'loss:' + str(loss_np) + '.ckpt'
    savePath = saver.save(sess, modelName)
    print('Model saved at: ', savePath)
    print('Loss:', loss_np)
    sess.run(train_iterator.initializer)

writer.close();

## HOW TO VALIDATE:
# call as follows (if False used to skip this while still highlighting/syntax checking it)
if (False):
    sess.run(loss, feed_dict={handle: val_handle})

# figure out how to save weights and save them here
# print(loss_np)


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
#
