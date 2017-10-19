#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

#include <iostream>
#include <cuda.h>

using namespace tensorflow;
using namespace std;
using namespace shape_inference;

Status ShapeFn(InferenceContext* c)
{
	//check input shape has 4 dimensions (batch, width, height, channels)
	ShapeHandle a_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &a_shape));

	//check indices has 4 dimensions (width, height, channel_in, channel_out)
	ShapeHandle b_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &b_shape));

	if (c->Value(c->Dim(a_shape,3)) != c->Value(c->Dim(b_shape,2)))
			return errors::InvalidArgument(
			"a and b 3rd dimension must be matched");

	DimensionHandle batch_size_dim = c->Dim(a_shape, 0);
	DimensionHandle in_rows_dim = c->Dim(a_shape, 1);
	DimensionHandle in_cols_dim = c->Dim(a_shape, 2);

	DimensionHandle filter_rows_dim = c->Dim(b_shape, 0);
	DimensionHandle filter_cols_dim = c->Dim(b_shape, 1);
	DimensionHandle input_depth = c->Dim(b_shape, 2);
	DimensionHandle depth_multiplier = c->Dim(b_shape, 3);
	
	std::vector<int32> strides;
	int32 stride_rows;
  	int32 stride_cols;
  	TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  	stride_rows = strides[1];
    stride_cols = strides[2];

	DimensionHandle output_rows, output_cols;
	c->Divide(in_rows_dim, stride_rows, false, &output_rows);
	if (c->Value(c->Dim(a_shape,1)) % stride_rows != 0)
		c->Add(output_rows, 1, &output_rows);
	c->Divide(in_cols_dim, stride_cols, false, &output_cols);
	if (c->Value(c->Dim(a_shape,2)) % stride_cols != 0)
		c->Add(output_cols, 1, &output_cols);
	// TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
	// 	c, in_rows_dim, filter_rows_dim, stride_rows, "SAME", &output_rows));
	// TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
	// 	c, in_cols_dim, filter_cols_dim, stride_cols, "SAME", &output_cols));

	ShapeHandle output_shape;
	output_shape =
        c->MakeShape({batch_size_dim, output_rows, output_cols, depth_multiplier});

	//set output size
	c->set_output(0, output_shape);  // (batch, channels, width, height)

	return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("CustomConvolution")
		.Input("a: T")
		.Input("b: T")
		.Attr("debug_mode: bool=false")
		.Attr("method: int=0")
		.Attr("strides: list(int)=[1,1,1,1]")
		.Output("c: T")
		.Attr("T: {int32, float32, float64}")
		.SetShapeFn(ShapeFn);

//declare kernel launcher
template <typename dtype>
void launchAddKernel(const dtype* a, const dtype* b, dtype* c, int kernel_size, 
	                 int input_channel, int output_channel, int kernel_len, 
	                 int batch, int height, int width, std::vector<int32>  strides_, int method, bool debug);

template <typename dtype>
class CustomAddOp : public OpKernel {
public:
	explicit CustomAddOp(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		OP_REQUIRES_OK(context,
                   context->GetAttr("debug_mode", &debug_mode_));
		OP_REQUIRES_OK(context,
                   context->GetAttr("method", &method_));
		OP_REQUIRES_OK(context,
                   context->GetAttr("strides", &strides_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& a_tensor = context->input(0);
		const Tensor& k_tensor = context->input(1);
		const TensorShape& a_shape = a_tensor.shape();
		const TensorShape& k_shape = k_tensor.shape();
				
		//flatten tensors
		auto a_flat = a_tensor.flat<dtype>();
		auto k_flat = k_tensor.flat<dtype>();
		if (debug_mode_) printf("shape: %d, %d\n", a_flat.size(),k_flat.size());
		OP_REQUIRES(context,
			(int)k_shape.dim_size(0) == (int)k_shape.dim_size(1),
			errors::InvalidArgument("Kernel height and width should be same"));

		//allocate the output
		Tensor* output_tensor = nullptr;
		TensorShape output_shape;

		kernel_size_ = (int)k_shape.dim_size(0);
		input_channel_ = (int)a_shape.dim_size(3);
		output_channel_ = (int)k_shape.dim_size(3);

		int output_rows, output_cols;

		output_rows = a_shape.dim_size(1) % strides_[1] == 0 ? a_shape.dim_size(1) / strides_[1] : a_shape.dim_size(1) / strides_[1] + 1;
		output_cols = a_shape.dim_size(2) % strides_[2] == 0 ? a_shape.dim_size(2) / strides_[2] : a_shape.dim_size(2) / strides_[2] + 1;

		output_shape.AddDim(a_shape.dim_size(0)); // batch
		output_shape.AddDim(output_rows); // height
		output_shape.AddDim(output_cols); // width 
		output_shape.AddDim(k_shape.dim_size(3)); // channel 
		OP_REQUIRES_OK(context,
			context->allocate_output(0,
			output_shape,&output_tensor));

		//get flat version to fill
		auto output = output_tensor->flat<dtype>();
		if (debug_mode_) printf("output: %d\n", output.size());

		// Call the cuda kernel launcher
		launchAddKernel<dtype>(a_flat.data(), k_flat.data(), output.data(), 
			kernel_size_, input_channel_, output_channel_, k_flat.size(), 
			output_shape.dim_size(0),(int)a_shape.dim_size(1),(int)a_shape.dim_size(2), 
			strides_, method_, debug_mode_);
	}
private:
	int input_channel_, output_channel_, kernel_size_, method_;
	std::vector<int32> strides_; 
	bool debug_mode_;
	float nnz_ratio_;
};

//register kernel with types needed
#define REGISTER_KERNEL(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CustomConvolution") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		CustomAddOp<type>) \

REGISTER_KERNEL(int);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL