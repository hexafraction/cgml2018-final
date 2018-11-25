#!/bin/python3.5.2
#Ostap Voynarovskiy and Andrey Akhmetov
#CGML Final
#Nov 16 2018
#Professor Curro

import os
import sys
import glob
import numpy as np
import stempeg
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from tqdm import tqdm

# Given to us in Wave-U-Net
BATCH_SIZE = 10
NUM_ITER = 4000

# Tell it what gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 



################ Adding My Stuff Below This ################
def loadWeights(file):
    #figure out how to load weights and push them into the model
    return True
def saveWeights(weights):
    #figure out how to extract weights and push them into a file
    return True

def WaveUNet(features):
    # Im making the model do nothing for now so we can debug the workflow
    #features=tf.expand_dims(features,3)#thought this would fix things... it didn't conv 1d needs a tensor of size 3 not 4 so the extra dim just fucked things
    print("############################")
    # Handle upsampling linearly as defined in model M4
    def UpSample(dataIn):
        dataIn = tf.expand_dims(dataIn, axis=1) 
        upsampled = tf.image.resize_bilinear(dataIn, [1, dataIn.get_shape().as_list()[2]*2]) 
        return upsampled
    
    # PARAMETERS OF THE MODEL
    convFilters = 24 # 15 not kernels so idk what to do with this now
    convStride = 1
    convPadding = 'valid'   # 'valid' means none (switch to valid at some point
    LAYERS = 12
    down = []
    current_layer = features
    down_kernel_size =15
    up_kernel_size = 5 #?
    #garbage temp model to get everything running
    print('shape of features ', features.shape) #features are the input
    
    l1 = features
    for i in range(LAYERS):
        #perform 1d Conv
        l1 = tf.layers.conv1d(l1,convFilters*(i+1),down_kernel_size,padding = convPadding)
        print("post conv 1d \t", l1.shape)
        down.append(l1)
        
        #downsample
        l1 = l1[:,::2,:]
        print("l1d \t\t", l1.shape)

    for i in reversed( range(LAYERS)):
        #upsampling
        l1 =UpSample(l1)
        #print('presqueeze \t',l1.shape)
        l1 = tf.squeeze(l1,1)
        #print('postsqueeze\t',l1.shape)
        l1 = l1[:,:-1,:] #exclude the last one to do the linear upsampling with an odd output 
        #print('postslice\t',l1.shape)
        l1 = tf.layers.conv1d(l1,convFilters*(i+1),up_kernel_size,padding = convPadding)
        print(l1.shape)#17,288
        #CROP AND CONCAT
        offset = int(int(down[i].shape[1]-l1.shape[1])/2)
        l1 = tf.concat([l1,down[i][:,offset:-offset,:]],2)
        print( 'concatenated', l1.shape)

    #for i in range(len(down)):
        #print(down[i].shape)
    # Shaping to output dimention
    fin = tf.layers.conv1d(l1,2,1)
    print('final layer \t', fin.shape)
    
    print("############################")
    #how to reduce the dimention later on to one you need for the output.
    predictions = fin
    # This is where we build the real model
    '''
    for i in range(LAYERS):
        #the going down part
        current_layer = tf.layers.conv1d(features,downconvFilters+(downconvFilters*i),convStride,convPadding)
        down.append(current_layer)
        current_layer = current_layer[:,::2,:] 
            
    # middle part   
    current_layer = tf.layers.conv1d(features,downconvFilters+(downconvFilters*i),convStride,convPadding)

    for j in range(LAYERS):
        #the going up part
        current_layer= UpSample(down[i-j])
        current_layer = tf.layers.conv1d(features,upconvFilters+(upconvFilters*i),convStride,convPadding)
    '''

    return predictions


sess = tf.Session()

rootpath = os.getcwd()
trainglob = os.path.join(rootpath, 'train', '*.tfrecord')
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

shape = [-1, 98291, 2] #[numsongs, numsamples per song, num channels]

mix = tf.decode_raw(tfrecord_features['mix'], tf.float32)
mix = tf.reshape(mix, shape)
#mix = sess.run(mix) #this initialization using sess fixed the graph problem where it was running out of data or whatever

drums = tf.decode_raw(tfrecord_features['drums'], tf.float32)
drums = tf.reshape(drums, shape)
#drums = sess.run(drums)

bass = tf.decode_raw(tfrecord_features['bass'], tf.float32)
bass = tf.reshape(bass, shape) 
#bass = sess.run(bass)

accomp = tf.decode_raw(tfrecord_features['accomp'], tf.float32)
accomp = tf.reshape(accomp, shape)
#accomp = sess.run(accomp)

vocals = tf.decode_raw(tfrecord_features['vocals'], tf.float32)
vocals = tf.reshape(vocals, shape)
#vocals = sess.run(vocals)

features  = mix #tf.placeholder(tf.float32, [None,16384,2])  # Should get batch size by 2 array of labels
labels = vocals #tf.placeholder(tf.float32, [None,16384,2])     # Revisit this idk if it's right

labels_predicted = WaveUNet(features) #this needs to be moved lower
print('num_params',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
loss = tf.losses.mean_squared_error(labels,labels_predicted)
# old loss from old project. Left as reference you can remove 
#tf.losses.sigmoid_cross_entropy(tf.stack([labels, 1-labels], 1),tf.squeeze(tf.stack([labels_predicted, -labels_predicted], 1))) \
#      + l*tf.reduce_sum([tf.nn.l2_loss(tV) for tV in tf.trainable_variables()])
#loss  = tf.reduce_mean(tf.pow(y-y_hat, 2)/2) #loss funtion = cross entropy + L2 norm

lr = .1
optim = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(loss)
#lets hope this works first. The paper gives slightly different parameters which we will update to later

init  = tf.global_variables_initializer()

sess.run(init)



for k in tqdm(range(0, NUM_ITER)):
    #x_np, labels_np = data.get_batch() # no more data.getBatch we use the tf records now
    loss_np, yhats, _ = sess.run([loss, labels_predicted, optim]) #, {features:mix,labels:vocals}
    if k%4000 == 0:
        print(loss_np)
print(loss_np)
#figure out how to save weights and save them here
#print(loss_np)






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



