#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void L2Normalize(const int n, const Dtype* in, Dtype* out, int length){
		CUDA_KERNEL_LOOP(index, n) {
			Dtype sum = 0;
			for ( int i = 0; i < length; i++ )
			{
				sum = sum + in[ index*length + i ] * in[ index*length + i ];
			}
			sum = sqrt(sum)+1e-6;
			for ( int i = 0; i < length; i++ )
			{
				out[ index*length + i ] = in[ index*length + i ] / sum;
			}
		}
	}


	template <typename Dtype>
	void CenterProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[ 0 ]->gpu_data();
		Dtype* top_data = top[ 0 ]->mutable_gpu_data();
		Dtype* weight_writable = this->blobs_[ 0 ]->mutable_gpu_data();
		const Dtype* weight = this->blobs_[ 0 ]->gpu_data();
		if ( M_ == 1 ) {
			caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, ( Dtype )1.,
				weight, bottom_data, ( Dtype )0., top_data);
			if ( bias_term_ )
				caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[ 0 ],
				this->blobs_[ 1 ]->gpu_data(), top_data);
		}
		else {
			// Step 1: Normalize weight
			L2Normalize<Dtype> << <CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS >> >(N_, weight, weight_writable, K_);
			CUDA_POST_KERNEL_CHECK;
			/*Dtype* squared_data = squared_.mutable_gpu_data();
			caffe_gpu_powx(N_*K_, weight, Dtype(2), squared_data);
			Dtype normsqr;
			for (int i = 0; i<N_; ++i) {
			caffe_gpu_asum<Dtype>(K_, squared_data + i*K_, &normsqr);
			caffe_gpu_scale<Dtype>(K_, pow(normsqr, -0.5), weight + i*K_, weight_writable + i*K_);
			caffe_gpu_scale<Dtype>(K_, rescale_coeff_, weight + i*K_, weight_writable + i*K_);
			}*/
			// Step 2: Get projection
			caffe_gpu_gemm<Dtype>(CblasNoTrans,
				transpose_ ? CblasNoTrans : CblasTrans,
				M_, N_, K_, rescale_coeff_,
				bottom_data, weight, ( Dtype )0., top_data);
			if ( bias_term_ )
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, ( Dtype )1.,
				bias_multiplier_.gpu_data(),
				this->blobs_[ 1 ]->gpu_data(), ( Dtype )1., top_data);
		}
	}

	template <typename Dtype>
	void CenterProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if ( this->param_propagate_down_[ 0 ] ) {
			const Dtype* top_diff = top[ 0 ]->gpu_diff();
			const Dtype* bottom_data = bottom[ 0 ]->gpu_data();
			// Gradient with respect to weight
			if ( transpose_ ) {
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
					K_, N_, M_,
					rescale_coeff_, bottom_data, top_diff,
					( Dtype )1., this->blobs_[ 0 ]->mutable_gpu_diff());
			}
			else {
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
					N_, K_, M_,
					rescale_coeff_, top_diff, bottom_data,
					( Dtype )1., this->blobs_[ 0 ]->mutable_gpu_diff());
			}
		}
		if ( bias_term_ && this->param_propagate_down_[ 1 ] ) {
			const Dtype* top_diff = top[ 0 ]->gpu_diff();
			// Gradient with respect to bias
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, ( Dtype )1., top_diff,
				bias_multiplier_.gpu_data(), ( Dtype )1.,
				this->blobs_[ 1 ]->mutable_gpu_diff());
		}
		if ( propagate_down[ 0 ] ) {
			const Dtype* top_diff = top[ 0 ]->gpu_diff();
			// Gradient with respect to bottom data
			if ( transpose_ ) {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, K_, N_,
					rescale_coeff_, top_diff, this->blobs_[0]->gpu_data(),
					( Dtype )0., bottom[ 0 ]->mutable_gpu_diff());
			}
			else {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					M_, K_, N_,
					rescale_coeff_, top_diff, this->blobs_[0]->gpu_data(),
					( Dtype )0., bottom[ 0 ]->mutable_gpu_diff());
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(CenterProjectionLayer);

}  // namespace caffe
