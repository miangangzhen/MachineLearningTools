#!/usr/bin/env python3
#-*-coding=utf-8-*-
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import re
from matplotlib import pyplot as plt


# input data path
TRAIN_LABEL_PATH = "/home/lty/imageRecognition/train.txt"
TRAIN_IMAGE_PATH = "/home/lty/imageRecognition/train/"
TRAIN_LABEL_DICT = {}
VAL_LABEL_PATH = "/home/lty/imageRecognition/val.txt"
VAL_IMAGE_PATH = "/home/lty/imageRecognition/val/"
VAL_LABEL_DICT = {}


# functions help to build cnn
def weigth_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return initial

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# load image and label
with open(TRAIN_LABEL_PATH, "r") as f:
	lines = f.readlines()
for line in lines:
	arr = line.strip().split(" ")
	TRAIN_LABEL_DICT[arr[0]] = int(arr[1])

with open(VAL_LABEL_PATH, "r") as f:
	lines = f.readlines()
for line in lines:
	arr = line.strip().split(" ")
	VAL_LABEL_DICT[arr[0]] = int(arr[1])


trian_image_list = []
train_label_list = []
val_image_list = []
val_label_list = []
for filename, label in TRAIN_LABEL_DICT.items():
	if os.path.isfile(os.path.join(TRAIN_IMAGE_PATH, filename)):
		trian_image_list.append(os.path.join(TRAIN_IMAGE_PATH, filename))
		train_label_list.append(label)

for filename, label in VAL_LABEL_DICT.items():
	if os.path.isfile(os.path.join(VAL_IMAGE_PATH, filename)):
		val_image_list.append(os.path.join(VAL_IMAGE_PATH, filename))
		val_label_list.append(label)

# queue capacity
CAPACITY = len(train_label_list)
VAL_COUNT = len(val_label_list)

# convert list to tensor
train_image_list_tensor = tf.convert_to_tensor(trian_image_list)
train_label_list_tensor = tf.convert_to_tensor(train_label_list)
val_image_list_tensor = tf.convert_to_tensor(val_image_list)
val_label_list_tensor = tf.convert_to_tensor(val_label_list)

# convert label to one hot vector like: 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
train_label_list_tensor = tf.one_hot(train_label_list_tensor,10)
val_label_list_tensor = tf.one_hot(val_label_list_tensor, 10)

# init queue to load data from file system
train_input_queue = tf.train.slice_input_producer(
	[train_image_list_tensor, train_label_list_tensor],
	shuffle=False)

val_input_queue = tf.train.slice_input_producer(
	[val_image_list_tensor, val_label_list_tensor],
	shuffle=False)

# load image from image list one by one
val_file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(val_file_content, channels=1)
train_image = tf.to_float(train_image)
# normalize
tmp = tf.reshape(train_image, [784])
tmp = (tmp - 0.0) / (255.0)
train_image = tf.reshape(tmp, [28, 28, 1])
train_image.set_shape((28, 28, 1))

# load label from label list one by one
train_label = train_input_queue[1]

# generate batch tensor, provide batch of image data and label
batch_size = 50
num_preprocess_threads = 1
min_queue_examples = 256
images = tf.train.shuffle_batch(
	[train_image, train_label],
	batch_size=batch_size,
	num_threads=num_preprocess_threads,
	capacity=CAPACITY,
	min_after_dequeue=min_queue_examples)


# do convert as above with validate data
val_file_content = tf.read_file(val_input_queue[0])
val_image = tf.image.decode_jpeg(val_file_content, channels=1)
val_image = tf.to_float(val_image)

tmp = tf.reshape(val_image, [784])
tmp = (tmp - 0.0) / (255.0)
val_image = tf.reshape(tmp, [28, 28, 1])
val_image.set_shape((28, 28, 1))

val_label = val_input_queue[1]


# define network
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# conv1
W_conv1 = weigth_variable([5, 5, 1, 32], "w_conv1")
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv2
W_conv2 = weigth_variable([5, 5, 32, 64], "w_conv2")
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connect layer1
W_fc1 = weigth_variable([7 * 7 * 64, 1024], "w_fc1")
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# full connect layer2
W_fc2 = weigth_variable([1024, 10], "w_fc2")
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# define entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define accuracy evaluate
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# define init op and saver op
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
merged = tf.summary.merge_all()


with tf.Session() as sess:

	# tensorboard	
	summary_writer = tf.summary.FileWriter('/tmp/tensorboard_logs', sess.graph)

	try:
		saver.restore(sess, "/tmp/imageModel-0")
		print("model restored.")
	except Exception as e:
		sess.run(init_op)
		exit()

	# Start populating the filename queue.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)


	print("###############################")

	# training
	for i in range(100):
		sample, label = sess.run([images[0], images[1]])
		if i % 100 == 0:
			#### to see the picture
			# tmp = sample[0].reshape([28, 28])
			# print(label[0])
			# im = Image.fromarray(np.uint8(tmp))
			# plt.imshow(im)
			# plt.show()
			####
			train_accuracy, summary = sess.run([accuracy, merged], feed_dict={x: sample, y_: label, keep_prob:1.0})
			
			print("step %d, training accuracy %g" % (i, train_accuracy))
			summary_writer.add_summary(summary, i)

		train_step.run(feed_dict={x: sample, y_: label, keep_prob:0.8})

	# evaluating
	print("start evaluate")
	sum_total = 0.0
	for i in range(VAL_COUNT):
		image, label = sess.run([val_image, val_label])
		sum_total += sess.run(accuracy, feed_dict={x: [image], y_: [label], keep_prob:1.0})
	
	print("eval precision:")
	print(sum_total / VAL_COUNT)

	# save model
	save_path = saver.save(sess, "/tmp/imageModel", global_step=0)
	print("save path:")
	print(save_path)
	print("###############################")

	coord.request_stop()
	coord.join(threads)