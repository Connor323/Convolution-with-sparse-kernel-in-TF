import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import numpy as np
from scipy.sparse import random
import time


try:
	_tutorial = tf.load_op_library('./build/libconv_sparse.so')
except Exception as e:
	_tutorial = tf.load_op_library('./libconv_sparse.so')
custom_convolution = _tutorial.custom_convolution

input_channel = 32
output_channel =64
kernel_size = 11
batch = 1
stride = 2
tolerance = 0.001
ratio  = 1
img_size = (256/ratio , 512/ratio )
input_random = True

a_shape = (batch,img_size[0],img_size[1],input_channel) # batch * height * width * ch_in 
b_shape = (kernel_size,kernel_size,input_channel,output_channel) # k_h * k_w * ch_in * ch_out

if input_random:
	a_data = np.random.random(a_shape).astype(np.float32)
else:
	a_data = np.ones(a_shape, np.float32)
	a_data[0, :, :, 0] *= 100
	a_data[0, :, :, 1] *= 50
	a_data[0, :, :, 2] *= 10

if input_random:
	b_data = np.reshape(random(1, np.prod(b_shape), density=0.10, dtype=np.float32).A, b_shape)
else:
	b_data = np.zeros(b_shape, np.float32)
	val = 1
	for i in range(output_channel):
		for j in range(input_channel):
			b_data[kernel_size/2-1, kernel_size/2-1, j, i] = val
			val += 1

a = tf.constant(a_data, shape=(a_shape[0], a_shape[1], a_shape[2], a_shape[3]), name="a")
b = tf.constant(b_data, shape=(b_shape[0], b_shape[1], b_shape[2], b_shape[3]), name="b")

c_cust = custom_convolution(a, b, debug_mode=True, method=0, strides=[1,stride,stride,1])
c = tf.nn.conv2d(a, b, padding="SAME", strides=[1, stride, stride, 1])

with tf.Session() as sess:
	for _ in range(1):
		start_time = time.time()
		expected = np.array(sess.run(c))
		print "time buildin: %.3f" % ((time.time() - start_time) * 1000)

		start_time = time.time()
		result = sess.run(c_cust)
		print "time costumized: %.3f" % ((time.time() - start_time) * 1000)
		print "result: ", result.shape, "expected: ", expected.shape
		# print "result", result
		# print "expected", expected
		print "is calculation correct? ", (abs(result - expected) < tolerance).all()
# diff = (abs(result - expected) > tolerance).astype(int) * 200

# cv2.imwrite("img_expected.png", expected[0, ...].astype(np.uint8))
# cv2.imwrite("img_result.png", result[0, ...].astype(np.uint8))
# cv2.imwrite("img_diff.png", diff[0, ...].astype(np.uint8))
