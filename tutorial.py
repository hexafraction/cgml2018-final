from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(folder_path, one_hot=True)

loss = tf.losses.softmax_cross_entropy(labels, output)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Open the session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(steps):
	# Get the next batch
	input_batch, labels_batch = next_batch(100)
	feed_dict = {x_input: input_batch, y_labels: labels_batch}

	# Print the current batch accuracy every 100 steps
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict=feed_dict)
		print("Step %d, training batch accuracy %g"%(i, train_accuracy))

	# Run the optimization step
	train_step.run(feed_dict=feed_dict)

# Print the test accuracy once the training is over
print("Test accuracy: %g"%accuracy.eval(feed_dict={x_input: test_images, y_labels: test_labels}))

input = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

output = tf.layers.dense(inputs=input, units=labels_size)

hidden = tf.layers.dense(inputs=input, units=1024, activation=tf.nn.relu)
output = tf.layers.dense(inputs=hidden, units=labels_size)

input2d = tf.reshape(input, [-1,image_size,image_size,1])

conv1 = tf.layers.conv2d(inputs=input2d, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
pool_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])
hidden = tf.layers.dense(inputs= pool_flat, units=1024, activation=tf.nn.relu)
output = tf.layers.dense(inputs=hidden, units=labels_size)

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

should_drop = tf.placeholder(tf.bool)

hidden = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=hidden, rate=0.5, training=should_drop)
output = tf.layers.dense(inputs=dropout, units=labels_size)