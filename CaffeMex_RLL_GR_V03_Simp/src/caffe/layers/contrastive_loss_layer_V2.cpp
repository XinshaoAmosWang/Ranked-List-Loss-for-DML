#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer_V2.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveLossV2Layer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  
  
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  //diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  
  // vector of ones used to sum along channels
  /*
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  */
  negative_threshold = this->layer_param_.contrastive_loss_param_v2().negative_threshold();
  legacy_version =
      this->layer_param_.contrastive_loss_param_v2().legacy_version();
  pn_margin = this->layer_param_.contrastive_loss_param_v2().pn_margin();
}

template <typename Dtype>
void ContrastiveLossV2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i

  const int channels = bottom[0]->channels();
 
  
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
     
    // gradient L2 normalisation => unit gradient vector
    // Dtype(1.0)/sqrt(dist_sq_.cpu_data()[i])
    caffe_cpu_scale(channels, Dtype(1.0) / Dtype( sqrt( dist_sq_.cpu_data()[i] ) + Dtype(1e-6)), 
      diff_.cpu_data() + (i*channels), diff_.mutable_cpu_data() + (i*channels));
    
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs

      Dtype dist = std::max<Dtype>(sqrt(dist_sq_.cpu_data()[i]) - (negative_threshold - pn_margin),
          Dtype(0.0));

      loss += dist*dist;
    } 
    else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max(negative_threshold - dist_sq_.cpu_data()[i], Dtype(0.0));
      } 
      else {
        //default
        Dtype dist = std::max<Dtype>(negative_threshold - sqrt(dist_sq_.cpu_data()[i]),
          Dtype(0.0));
        loss += dist*dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];   //without nomalization

      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          /*
          Dtype mdist = sqrt(dist_sq_.cpu_data()[j]) - (negative_threshold - pn_margin);
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          }
          else{
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
          */
        } 
        else {  // dissimilar pairs
          Dtype beta(0.0);
          beta = -alpha;
          caffe_cpu_axpby(
              channels,
              beta,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          /*Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = negative_threshold - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            //default
            mdist = negative_threshold - sqrt(dist_sq_.cpu_data()[j]);
            beta = -alpha;
          }
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                beta,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }*/
        }// dissimiliar pairs
      }//j
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossV2Layer);
#endif

INSTANTIATE_CLASS(ContrastiveLossV2Layer);
REGISTER_LAYER_CLASS(ContrastiveLossV2);

}  // namespace caffe
