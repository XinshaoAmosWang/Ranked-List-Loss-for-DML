#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
  __global__ void UpsampleForward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Dtype* bottom_data,
      const Dtype* upsample_index_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int c = index / (in_w * in_h);
      int i = index - c * in_w * in_h;
      int j = c * out_w * out_h;
      int upsample_idx = static_cast<int>(upsample_index_data[i]);
      top_data[j + upsample_idx] = bottom_data[index];
    }
  }

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* upsample_index_data = upsample_index_.gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  int bottom_count = bottom[0]->count();
  UpsampleForward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count, width_, height_, upsample_w_, upsample_h_,
      bottom_data, upsample_index_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
  __global__ void UpsampleBackward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Dtype* top_diff,
      const Dtype* upsample_index_data, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int c = index / (in_w * in_h);
      int i = index - c * in_w * in_h;
      int j = c * out_w * out_h;
      int upsample_idx = static_cast<int>(upsample_index_data[i]);
      bottom_diff[index] = top_diff[j + upsample_idx];
    }
  }

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* upsample_index_data = upsample_index_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
    UpsampleBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, width_, height_, upsample_w_, upsample_h_,
        top_diff, upsample_index_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);


}  // namespace caffe
