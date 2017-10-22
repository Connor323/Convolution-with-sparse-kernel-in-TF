#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>
#include <typeinfo>
#include <stdexcept>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename dtype> __global__ 
void convolve_global (const dtype * __restrict__ Md, dtype * __restrict__ Rd, 
    int width, int height, int output_rows, int output_cols, int strides_X, int strides_Y, 
    int kernel_size, int channels, int *nnzs, int *start_idxs, int output_channel, 
    int batch_offset_in, int batch_offset_out, 
    const dtype * __restrict__ cooVal_Kdc, const int * __restrict__ cooRow_Kdc, 
    const int * __restrict__ cooCol_Kdc, const int * __restrict__ cooDep_Kdc){

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row_Md = row * strides_X;
    int col_Md = col * strides_Y;
    int ch_out = blockIdx.z*blockDim.z + threadIdx.z;
    
    if(row < output_rows  &&  col < output_cols && ch_out < output_channel &&
       row_Md < height && col_Md < width){

        dtype sum = 0;
        int nnz = nnzs[ch_out];
        int start_idx = start_idxs[ch_out];
        int pixel;
        int local_pixel;
        int working_pixel;

    	dtype val = 0;
    	pixel = batch_offset_out + ch_out + row*output_cols*output_channel + col*output_channel;
        for(int i=start_idx; i < start_idx + nnz; i++){
            int y = cooRow_Kdc[i] - kernel_size/2, x = cooCol_Kdc[i] - kernel_size/2, z = cooDep_Kdc[i];
            local_pixel = z + row_Md*width*channels + col_Md*channels + batch_offset_in;
            working_pixel = local_pixel + x*channels + y*width*channels;
            if (y + row_Md < 0 || y + row_Md >= height || x + col_Md < 0 || x + col_Md >= width) continue;

            sum = (dtype)Md[working_pixel] * cooVal_Kdc[i];
            val += sum;
        }
    	Rd[pixel] = val;
    }
}

int calculateBlockSize(int width, int height, int *max_nnz, int slots_per_sm, int registers_per_thread){ // TODO: read more...

    //Getting the properties of the device
    int device;
    cudaDeviceProp properties;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);

    int max_threads_sm = properties.maxThreadsPerMultiProcessor;    //2048
    int max_block_size = properties.maxThreadsPerBlock; //1024
    int max_registers_sm = properties.regsPerBlock; //65536

    //Calculating the size
    int max_concurrent_threads = max_threads_sm*registers_per_thread <= max_registers_sm ? max_threads_sm : max_registers_sm/registers_per_thread;    //that is the max number of threads I could ever execute at the same time (either because of the number of thread slots or the number of registers)
    float sm_threads = 0, sm_threads_aux;  //number of concurrent threads being executed in the same SM
    int slot_size;
    int slots;
    int block_size, block_size_aux;
    int total_threads;
    int total_blocks;
    float batches, batches_aux;  //number of times the SM will be loaded with threads

    //Maximize the number of warps
    for(int i=1; i<=slots_per_sm; i++){
        slot_size = max_concurrent_threads/i;
        if(slot_size <= max_block_size){
            block_size_aux = (int) sqrt(slot_size); //I'm looking for square kernels
            sm_threads_aux = i*block_size_aux*block_size_aux;
            
            total_blocks = (width/block_size_aux) * (height/block_size_aux);

            if(width%block_size_aux != 0)
                total_blocks += height/block_size_aux;
            if(height%block_size_aux != 0)
                total_blocks += width/block_size_aux;
            if(width%block_size_aux != 0  &&  height%block_size_aux != 0)
                total_blocks--;    //overlapping

            total_threads = total_blocks * block_size_aux*block_size_aux;
            batches_aux = (float)total_threads/sm_threads_aux;

            if(sm_threads_aux > sm_threads  ||  batches_aux < batches){ //always looking for the greater number of concurrent threads, avoiding the underutilization
                sm_threads = sm_threads_aux;
                batches = batches_aux;
                block_size = block_size_aux;
                slots = i;
            }
        }
    }
    return block_size;
}

template <typename dtype>
void ImageConvolution(const dtype * __restrict__ M, dtype * cooVal_Kdc, int * cooRow_Kdc, int * cooCol_Kdc, int * cooDep_Kdc, int *nnz, 
                      dtype * R, int width, int height, int input_channel, int kernel_size, int output_channel, 
                    int batch, std::vector<int>  strides, int *start_idx, int *max_nnz, int method, bool debug_mode){
    //Grid and Block dim
    int max_threads_xy = 1024;
    int max_threads_z = 64;
    dim3 dimGrid, dimBlock;
    int output_rows, output_cols;

    output_rows = height % strides[1] == 0 ? height / strides[1] : height / strides[1] + 1;
    output_cols = width % strides[2] == 0 ? width / strides[2] : width / strides[2] + 1;

    int size_per_batch_input = height * width * input_channel;
    int size_per_batch_output = output_rows * output_cols * output_channel;

    //Time measuring
    cudaEvent_t start, stop;
    float time;
    //Invocation to convolve
    dimBlock.x = min(min(10, output_cols), max_threads_xy);
    dimBlock.y = min(min(10, output_rows), max_threads_xy);
    dimBlock.z = min(min(10, output_channel), max_threads_z);
    dimGrid.x = ceil((float)output_cols / dimBlock.x);
    dimGrid.y = ceil((float)output_rows / dimBlock.y);
    dimGrid.z = ceil((float)output_channel / dimBlock.z);
           
    if (debug_mode) printf("%d, %d, %d, %d, %d, %d\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);

    //Calculating the shared memory size
    // const int shared_mem_size = (dimBlock.x*dimBlock.y + \
    //                                 dimBlock.x*(kernel_size/2)*2 + \
    //                                 dimBlock.y*(kernel_size/2)*2 + \
    //                                 4*(kernel_size/2)*(kernel_size/2)) * sizeof(dtype);

    // if (debug_mode && method == 1) printf("shared_mem_size: %d\n", shared_mem_size); 
    // printf("size of scalar: %d\n", sizeof(dtype));

    //Preparing time measuring
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start,0)); 
    
    //Calling the kernel
    for(int b=0; b<batch; b++){
        int batch_offset_in = b * size_per_batch_input;
        int batch_offset_out = b * size_per_batch_output;
        if (method == 0)
            convolve_global<dtype><<<dimGrid,dimBlock>>>(M,R,width,height,output_rows,output_cols,
                                                 strides[1],strides[2],kernel_size,
    	                                         input_channel,nnz,start_idx,output_channel, 
                                                 batch_offset_in, batch_offset_out,
                                                 cooVal_Kdc,cooRow_Kdc,cooCol_Kdc,cooDep_Kdc);
        else{
            throw std::invalid_argument("not support shared memory version for now\n");
            // convolve_shared<dtype><<<dimGrid,dimBlock,shared_mem_size>>>(M,R,width,height,kernel_size,
            //                                      input_channel,nnz,start_idx,output_channel,
            //                                      batch_offset_in, batch_offset_out,dimGrid,shared_mem_size, 
            //                                      cooVal_Kdc,cooRow_Kdc,cooCol_Kdc,cooDep_Kdc);
        }
        
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
   	}
    gpuErrchk(cudaEventRecord(stop,0));
    gpuErrchk(cudaEventSynchronize(stop))
    gpuErrchk(cudaEventElapsedTime(&time,start,stop));

    if (debug_mode) std::cout << time <<std::endl;
}

template <typename dtype> __global__ 
void prepare_kernel(const dtype * __restrict__ kernel, int kernel_len, 
                            int output_channel, int *__restrict__ nnz, int size_each_kernel){
    dtype val;
    for(int i=0; i<size_each_kernel; i++){
        int ch = blockIdx.x*blockDim.x + threadIdx.x;
        val = kernel[i * output_channel + ch];
        if(abs(val) > 0.000001){
            nnz[ch] = nnz[ch] + 1;
        }  
    }
}

template <typename dtype> __global__ 
void process_sparse_kernel(const dtype * __restrict__ kernel, 
                           int kernel_len, int kernel_size, int output_channel, int input_channel, int *start_idx, 
                           dtype *cooVal_Kdc, int *cooRow_Kdc, int *cooCol_Kdc, int *cooDep_Kdc, int *nnz, int *outch_pts,
                           int size_each_kernel){
    int index;
	
    for(int i=0; i<size_each_kernel; i++){
        int ch = blockIdx.x*blockDim.x + threadIdx.x;
        index = i * output_channel + ch;
        if(abs(kernel[index]) > 0.000001){
            cooVal_Kdc[start_idx[ch] + outch_pts[ch]] = kernel[index];
            cooRow_Kdc[start_idx[ch] + outch_pts[ch]] = index / (kernel_size * input_channel * output_channel);
            cooCol_Kdc[start_idx[ch] + outch_pts[ch]] = index % (kernel_size * input_channel * output_channel) / (input_channel * output_channel);
            cooDep_Kdc[start_idx[ch] + outch_pts[ch]] = index % (input_channel * output_channel) / output_channel;
            outch_pts[ch]++;
        }
    }
}

__global__ 
void post_process_sumval(int output_channel, int *nnz, int *start_idx, int *total_nnz, int *max_nnz, bool print_info){
    max_nnz[0] = 0;
    total_nnz[0] = 0;
    for(int i=0; i<output_channel; i++){
    	start_idx[i] = i == 0 ? 0 : start_idx[i-1] + nnz[i-1];
        total_nnz[0] = total_nnz[0] + nnz[i];
        if(nnz[i] > max_nnz[0]){
            max_nnz[0] = nnz[i];
        }
    	if (print_info)
    		printf("Kernel: %d,, nnz: %d\n", i, nnz[i]);
	}
}

__global__
void set_all_zero(int *nnz, int *outch_pts){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    nnz[i] = 0;
    outch_pts[i] = 0;
}

__global__
void set_zero(int *total_nnz){
    total_nnz[0] = 0;
}

template <typename dtype>
void launchAddKernel(const dtype* activation, const dtype* kernel, dtype* output, 
	                 int kernel_size, int input_channel, int output_channel, int kernel_len,
	                 int batch, int height, int width, std::vector<int>  strides, 
                     int method, bool debug_mode) {
	// convert dense tensor to sparse in COO format
    int *nnz, *total_nnz, *start_idx, *max_nnz;
	int * outch_pts;
    if(debug_mode) printf("type: %s\n", typeid(dtype).name());
	gpuErrchk(cudaMallocManaged(&nnz, output_channel*sizeof(int)));
    gpuErrchk(cudaMallocManaged(&total_nnz, sizeof(int)));
	gpuErrchk(cudaMallocManaged(&max_nnz, sizeof(int)));
	gpuErrchk(cudaMallocManaged(&start_idx, output_channel*sizeof(int)));
	gpuErrchk(cudaMallocManaged(&outch_pts, output_channel*sizeof(int)));

    set_zero<<<1, 1>>>(total_nnz);
    
    // count nnz
    dim3 dimGrid, dimBlock;
    int size_each_kernel = kernel_len / output_channel;
    dimBlock.x = min(1024, output_channel);
    dimGrid.x = ceil((float)output_channel / dimBlock.x);

    set_all_zero<<<dimGrid, dimBlock>>>(nnz, outch_pts);
	prepare_kernel<dtype><<<dimGrid, dimBlock>>>(kernel, kernel_len, output_channel, nnz, size_each_kernel);
	cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
	
	post_process_sumval<<<1, 1>>>(output_channel, nnz, start_idx, total_nnz, max_nnz, false);
	cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    if (debug_mode) printf("total_nnz: %d\n", total_nnz[0]);

    dtype * cooVal_Kdc;
    int * cooRow_Kdc;
    int * cooCol_Kdc;
    int * cooDep_Kdc;
    gpuErrchk(cudaMalloc((void**) &cooVal_Kdc, total_nnz[0]*sizeof(dtype)));
    gpuErrchk(cudaMalloc((void**) &cooRow_Kdc, total_nnz[0]*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &cooCol_Kdc, total_nnz[0]*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &cooDep_Kdc, total_nnz[0]*sizeof(int)));

    process_sparse_kernel<dtype><<<dimGrid, dimBlock>>>(kernel, kernel_len, kernel_size, output_channel, input_channel, start_idx, 
    	                                   cooVal_Kdc, cooRow_Kdc, cooCol_Kdc, cooDep_Kdc, nnz, outch_pts, size_each_kernel);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

	ImageConvolution<dtype>(activation, cooVal_Kdc, cooRow_Kdc, cooCol_Kdc, cooDep_Kdc,
	                  nnz, output, width, height, input_channel, kernel_size, output_channel, batch, strides, 
                      start_idx, max_nnz, method, debug_mode);
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}


//forward declaration for all the types needed
#define ADD_KERNEL_TYPE(type)							\
	template void launchAddKernel<type>(				\
		const type* a, const type* b, type* c, int kernel_size, \
		int input_channel, int output_channel, int kernel_len, \
		int batch, int height, int width, std::vector<int>  strides, \
        int method, bool debug)	\

ADD_KERNEL_TYPE(int);
ADD_KERNEL_TYPE(float);
ADD_KERNEL_TYPE(double);

#undef ADD_KERNEL_TYPE