'''
This is the demo for loading and running sparse convolution op. 

'''

import tensorflow as tf
import numpy as np
from scipy.sparse import random
import time

# load op
try:
	_conv_sparse = tf.load_op_library('./build/libconv_sparse.so')
except Exception as e:
	_conv_sparse = tf.load_op_library('./libconv_sparse.so')
sparse_convolution = _conv_sparse.custom_convolution

# tensor setting
input_channel = 32
output_channel =64
kernel_size = 11
sparse_density = 0.1
batch = 1
stride = 1
tolerance = 0.001
input_size = (256 , 512 )

input_shape = (batch,input_size[0],input_size[1],input_channel) # batch * height * width * ch_in 
kernel_shape = (kernel_size,kernel_size,input_channel,output_channel) # k_h * k_w * ch_in * ch_out

input_data = np.random.random(input_shape).astype(np.float32)
kernel_data = np.reshape(random(1, np.prod(kernel_shape), density=sparse_density, dtype=np.float32).A, kernel_shape)

inputT = tf.constant(input_data, shape=input_shape, name="input")
kernelT = tf.constant(kernel_data, shape=kernel_shape, name="kernel")

result_Sparse = sparse_convolution(inputT, kernelT, debug_mode=False, method=0, strides=[1,stride,stride,1])
result_TF = tf.nn.conv2d(inputT, kernelT, padding="SAME", strides=[1, stride, stride, 1])

with tf.Session() as sess:
	start_time = time.time()
	expected = np.array(sess.run(result_TF))
	print "time buildin: %.3f" % ((time.time() - start_time) * 1000)

	start_time = time.time()
	result = sess.run(result_Sparse)
	print "time costumized: %.3f" % ((time.time() - start_time) * 1000)
	print "result shape: ", result.shape, "expected shape: ", expected.shape
	print "is calculation correct? ", (abs(result - expected) < tolerance).all()

