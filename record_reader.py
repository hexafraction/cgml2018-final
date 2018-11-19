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

# Tell it what gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 


################ Model is going above this ################
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
##########################################################################################

def WaveUNet(features,labels,mode):

	inputShape =  [batch, in_width, in_channels] 
	downconvFilters = 15
	upconvFilters = 5
	convStride = 1 
	padding = 'VALID'

	layer1 = tf.nn.conv1d(features,filters,stride,padding)
	d1 = 
	
	layer2 = tf.nn.conv1d(Input,filters,stride,padding)
	d2 = 
	
	layer3 = tf.nn.conv1d(Input,filters,stride,padding)
	d3 = 

	layer4 = tf.nn.conv1d(Input,filters,stride,padding)
	d4 = 
	
	layer5 = tf.nn.conv1d(Input,filters,stride,padding)
	d5 = 
	
	layer6 = tf.nn.conv1d(Input,filters,stride,padding)
	d6 = 
	
	layer7 = tf.nn.conv1d(Input,filters,stride,padding)
	d7 = 

	layer8 = tf.nn.conv1d(Input,filters,stride,padding)
	d8 = 

	layer9 = tf.nn.conv1d(Input,filters,stride,padding)
	d9 = 
	
	layer10 = tf.nn.conv1d(Input,filters,stride,padding)
	d10 = 

	layer11 = tf.nn.conv1d(Input,filters,stride,padding)
	d11 = 
	
	layer12 = tf.nn.conv1d(Input,filters,stride,padding)
	d12 = 
	
	layer13 = tf.nn.conv1d(Input,filters,stride,padding)
	d13 = 
	
	layer14 = tf.nn.conv1d(Input,filters,stride,padding)
	d14 = 

	layer15 = tf.nn.conv1d(Input,filters,stride,padding)
	d15 = 

	



#!/bin/python3.6
#Ostap Voynarovskiy
#CGML HW2
#Sept 19 2018
#Professor Curro

#from tensorflow.python import debug as tfdbg

BATCH_SIZE = 200
NUM_ITER = 4000 	# iterations of training 

class Data(object):
	def __init__(self):
		#create spirals
		nPoints = 200 
		self.index = np.arange(nPoints)
		self.nPoints = nPoints
		self.featx, self.featy, self.lab  = self.gen_spiral(nPoints)

	def gen_spiral(self,nPoints):
		scale = 1
		offset = 1
		sigma = .2

		t = np.linspace(0,3.5*np.pi,num = nPoints)
		noise0 = sigma*np.random.normal(size=nPoints)
		noise1 = sigma*np.random.normal(size=nPoints)
		noise2 = sigma*np.random.normal(size=nPoints)
		noise3 = sigma*np.random.normal(size=nPoints)
		
		#add normal noise
		theta0 = -t*scale + noise0
		r0 = (t + offset) + noise1
		theta1= -t*scale + np.pi + noise2	#the addition of pi does a 180 degree shift
		r1 = (t + offset) + noise3

		#convert from polar to cartesian
		self.x0 = np.cos(theta0)*(r0)
		self.y0 = np.sin(theta0)*(r0)
		cat0 = [0]*nPoints 			# the categories
		self.x1 = np.cos(theta1)*(r1) 
		self.y1 = np.sin(theta1)*(r1)
		cat1 = [1]*nPoints 			# the categories
		return np.concatenate((self.x0,self.x1)),np.concatenate((self.y0,self.y1)), np.concatenate((cat0,cat1)) 		

	def get_batch(self):
		choices = np.random.choice(self.nPoints*2, size=BATCH_SIZE)
		return list(zip(self.featx[choices],self.featy[choices])), self.lab[choices]


def f(x): #this is where we decide our tunable parameters and create our perceptron 
	m1 = 74	# first layer nodes = my fav 2 numbers 
	m2 = 47	# second layer nodes = my fav 2 numbers but swapped
	m3 = 1 	# one so that its a single yes or no

	# These are the initializations of the things we will learn including w's b's and 

	# Weight matricies should all be aproximately gaussian distribution since we care about 
	# diversity but wanna give all features similar chances on average. 
	w1 = tf.get_variable('w1', [2,m1], tf.float32,tf.random_normal_initializer()) 
	w2 = tf.get_variable('w2', [m1, m2], tf.float32,tf.random_normal_initializer()) 
	w3 = tf.get_variable('w3', [m2, m3], tf.float32,tf.random_normal_initializer()) 

	# start at 0
	b1 = tf.get_variable('b1', [1,m1], tf.float32, tf.random_normal_initializer())  #update
	b2 = tf.get_variable('b2', [1,m2], tf.float32, tf.random_normal_initializer()) 
	b3 = tf.get_variable('b3', [1,m3], tf.float32, tf.random_normal_initializer()) 

	#activation functions
	layer1 = tf.nn.elu(tf.matmul(x,w1)+b1)	# Activation function 1
	layer2 = tf.nn.leaky_relu(tf.matmul(layer1,w2)+b2)		# Activation function 2
	layer3 = (tf.matmul(layer2,w3)+b3)				# produce logits for cross entropy loss

	return _predicted_labels_

features  = tf.placeholder(tf.float32, [None,1]) 	# Should get batch size by 2 array of labels
labels = tf.placeholder(tf.float32, [None]) 		# Should get batch size by 1 array ...
														# we want a binary classification
labels_predicted = f(features)
#lr = tf.placeholder(tf.float32,shape=[]) #variable learning rate
# which w are we taking the norm of there are 3?
l = 0.002; # l is lambda 

loss = tf.losses.sigmoid_cross_entropy(tf.stack([labels, 1-labels], 1),tf.squeeze(tf.stack([labels_predicted, -labels_predicted], 1))) \
	   + l*tf.reduce_sum([tf.nn.l2_loss(tV) for tV in tf.trainable_variables()])
#loss  = tf.reduce_mean(tf.pow(y-y_hat, 2)/2) #loss funtion = cross entropy + L2 norm

lr = .1
optim = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss) #this does gradient descent

init  = tf.global_variables_initializer()
################ Model is going above this ################
sess = tf.Session()
#sess = tfdbg.LocalCLIDebugWrapperSession(sess)
sess.run(init)

data = Data()
for _ in tqdm(range(0, NUM_ITER)):
    x_np, labels_np = data.get_batch()
    loss_np, yhats, _ = sess.run([loss, labels_predicted, optim], feed_dict={features: x_np, labels: labels_np})

print(loss_np)
#rslt=sess.run(tf.stack(labels_predicted), feed_dict={features: list(zip(data.featx,data.featy))})
fig1= plt.figure(1)

xc,yc = np.linspace(-15,15,500),np.linspace(-15,15,500) 
xv,yv = np.meshgrid(xc,yc)

feat = np.array(list(zip(xv.flatten(),yv.flatten())))
res  = sess.run(labels_predicted, feed_dict={features: feat })  # lt = sess.run(what_you_want,    feed_dict={features: what_you_have})
cont = sess.run(tf.sigmoid(res))
plt.contourf(xv,yv,cont.reshape((500,500)),[0,.5,1])
plt.scatter(data.x0,data.y0,color='white')
plt.scatter(data.x1,data.y1,color='black')

w= plt.xlabel('x')
h= plt.ylabel('y')
h.set_rotation(0)


print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

plt.title("3 Layer Perceptron ")
plt.axis('equal') #make it so that it isnt warped
plt.show()


