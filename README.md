# Convolution-with-sparse-kernel-in-TF

**Still in progress...**

TensorFlow implementation of convolution with sparse kernel. This operator is initially for deep learning pruning based on Song Han's [work](https://arxiv.org/pdf/1506.02626.pdf). For now, this operator only supports TensorFlow with CUDA support.

## Usage: 
1. cmake .
2. make 
3. To import the TF customized op, do
```python
_conv_sparse = tf.load_op_library('path_to_source_file/libconv_sparse.so')
conv_op = _conv_sparse.custom_convolution
```
4. Parameters:
* **Input Tensor**: 4D tensor in int32, float32 and float64 with shape [batch, height, width, channel_in]
* **Kernel Tensor**: 4D tensor in int32, float32 and float64 with shape [k_height, k_width, channel_in, channel_out]
* **debug_mode**: boolean value; True if need to print information about tensor shape, computing time etc. 
* **method**: integer value with 0 or 1; 0 if using GPU global memory, 1 if using GPU shared memory (only support 0 for now).
* **strides**: list of integer [stride_batch, stride_h, stride_w, stride_ch]. (only support h, w strides for now)

## Results:
Following figures show the current performance with respect to different hyperparameters, including kernel size, input channel, output channel, ratio of NNZ (number of nonzero) and the size of input. The unit along the vertical axis is in ms and these results is from NVIDIA GeForce 940M. 

**Input: 1 * 256 * 512 * 3; Kernel: ? * ? * 3 * 64; Ratio of NNZ: 0.1**

![ ](./images/kernel.png)

**Input: 1 * 256 * 512 * ?; Kernel: 11 * 11 * ? * 3; Ratio of NNZ: 0.1**

![ ](./images/ch_in.png)

**Input: 1 * 256 * 512 * 3; Kernel: 11 * 11 * 3 * ?; Ratio of NNZ: 0.1**

![ ](./images/ch_out.png)

**Input: 1 * 256 * 512 * 32; Kernel: 11 * 11 * 32 * 64; Ratio of NNZ: ?**

![ ](./images/nnz.png)

**Input: 1 * ? * ? * 3; Kernel: 11 * 11 * 3 * 64; Ratio of NNZ: 0.1**

![ ](./images/image.png)

## Discussion:
This sparse convolution is much faster than the buildin dense convolution in TF in most of case, besides with larger ratio of NNZ (not sparse anymore) or the smaller size of input. Since the acceleration becomes more prominent when the size of input is larger (especially along height and width channels), the overall performance of a small input could be mediocre for now, but I will continue improving the proformance.   

## TODO:
 - [ ] Use shared memory for CUDA multithreading
 - [ ] Improve result precision (the precision for now is about 1e-3)
 - [ ] Add CPU support
 - [ ] Separate the computation of dense-to-sprase within this operation to speed up more (since we only need to convert once during inference)
 - [x] Fix the inconsistency when stride is over 2 (TF uses the [smallest padding scheme](https://www.tensorflow.org/api_guides/python/nn#Convolution))
 - [ ] Add gradient for training purpose

## Reference:
This work also refers to the work of [GPU convolution](https://github.com/fjmh89/convolution). 
