# Convolution-with-sparse-kernel-in-TF
Development a customized op in TensorFlow for convolution with sparse kernel

======
## Usage: 
1. cmake .
2. make 
3. To import the TF customized op, do
```python
_conv_sparse = tf.load_op_library('path_to_source_file/libconv_sparse.so')
conv_op = _conv_sparse.custom_convolution
```

======

**Still in progress...**
