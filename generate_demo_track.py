import math
import os
import sys

import numpy as np
import resampy
import soundfile as sf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import wavio
import tqdm

if len(sys.argv) < 5:
    print("USAGE: import22k.py checkpoint_name input_file outdir prefix")
    exit(-1)

BATCH_SIZE = 10
FRAGMENT_LENGTH = 65523
FRAGMENT_OFFSET = 14341

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

    for i in reversed(range(LAYERS - 1)):
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

mix = tf.placeholder(tf.float32, [None, 65523, 2])
features = mix

labels_predicted = WaveUNet(features)  # this needs to be moved lower

print('Num_params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

off = int(int(mix.shape[1] - labels_predicted.shape[1]) / 2)  # calculate how much needs to be cropped

saver = tf.train.Saver()

sess.__enter__()
saver.restore(sess, sys.argv[1])

print('Reading and resampling file...')
S, fs = sf.read(sys.argv[2])
S_resampled = resampy.resample(S, fs, 22050, 0)
nfrag = 1+(S_resampled.shape[0]-FRAGMENT_LENGTH)//FRAGMENT_OFFSET
lend = [FRAGMENT_OFFSET*i for i in range(nfrag)]
rend = [FRAGMENT_OFFSET*i+FRAGMENT_LENGTH for i in range(nfrag)]
bounds = list(zip(lend, rend))
print('Import complete. %d fragments of length %d, offset %d.' % (nfrag, FRAGMENT_LENGTH, FRAGMENT_OFFSET))

print("Performing inference...")
mc = mix[:, off:-off, :]
bigmix = [np.zeros([0, 2]), np.zeros([0, 2])]

for i in tqdm.tqdm(range(math.ceil(nfrag/BATCH_SIZE))):
    batchbounds = bounds[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    slices = [S_resampled[bnd[0]:bnd[1]] for bnd in batchbounds]
    stacked = np.stack(slices, 0)
    [lp, mx] = sess.run([labels_predicted, mc], feed_dict={mix: stacked})
    bigmix[0] = np.concatenate((bigmix[0], lp.reshape(-1, 2)), 0)
    bigmix[1] = np.concatenate((bigmix[1], mx.reshape(-1, 2)), 0)

wavio.write(os.path.join(sys.argv[3], sys.argv[4] + "_output.wav"), bigmix[0], 22050, sampwidth=3)
wavio.write(os.path.join(sys.argv[3], sys.argv[4] + "_input.wav"), bigmix[1], 22050, sampwidth=3)
