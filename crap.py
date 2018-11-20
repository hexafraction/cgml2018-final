# Down-convolution: Repeat strided conv
for i in range(self.num_layers):
    current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * i), self.filter_size, strides=1, activation=LeakyReLU, padding=self.padding) # out = in - filter + 1
    enc_outputs.append(current_layer)
    current_layer = current_layer[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1

current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * self.num_layers),self.filter_size,activation=LeakyReLU,padding=self.padding) # One more conv here since we need to compute features after last decimation

# Feature map here shall be X along one dimension

# Upconvolution
for i in range(self.num_layers):
    #UPSAMPLING
    current_layer = tf.expand_dims(current_layer, axis=1)

    if self.upsampling == 'learned':
        # Learned interpolation between two neighbouring time positions by using a convolution filter of width 2, and inserting the responses in the middle of the two respective inputs
        current_layer = Utils.learned_interpolation_layer(current_layer, self.padding, i)
    else:
        if self.context:
            current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
        else:
            current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1
    #current_layer = tf.layers.conv2d_transpose(current_layer, self.num_initial_filters + (16 * (self.num_layers-i-1)), [1, 15], strides=[1, 2], activation=LeakyReLU, padding='same') # output = input * stride + filter - stride
    current_layer = tf.squeeze(current_layer, axis=1)

    assert(enc_outputs[-i-1].get_shape().as_list()[1] == current_layer.get_shape().as_list()[1] or self.context) #No cropping should be necessary unless we are using context
    current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
    current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * (self.num_layers - i - 1)), self.merge_filter_size,
                                     activation=LeakyReLU,
                                     padding=self.padding)  # out = in - filter + 1

current_layer = Utils.crop_and_concat(input, current_layer, match_feature_dim=False)

#