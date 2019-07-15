#include <algorithm>
#include <cfloat>
#include <vector>
#include <boost/pointer_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/layers/distance_weighted_valid_pair_loss_V12_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DistanceWeightedValidPairLossV12Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //auxiliary layer
  LayerParameter contrastive_loss_param(this->layer_param_);
  contrastive_loss_param.set_type("ContrastiveLossV2");
  contrastive_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(contrastive_loss_param);

}

template <typename Dtype>
void DistanceWeightedValidPairLossV12Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::Reshape(bottom, top);
  //initialization of  pair data and pair lables, and auxiliary parameters
  sample_num = bottom[0]->num();
  feature_dim = bottom[0]->channels();
  //Valid construction of pairs
  pair_num = 0;
  pair_index_a.clear();
  pair_index_b.clear();
  for (int i = 0; i < sample_num - 1; i++){
    for(int j = i + 1; j < sample_num; j++){
      pair_num++;
      pair_index_a.push_back(i);
      pair_index_b.push_back(j);
    }
  }
  //
  vector<int> shape = bottom[0]->shape();
  shape[0] = pair_num;
  pair_data_a.Reshape(shape);
  pair_data_b.Reshape(shape);
  pair_label.Reshape(pair_num, 1, 1, 1);

  contrastive_loss_bottom.clear();
  contrastive_loss_bottom.push_back(&pair_data_a);
  contrastive_loss_bottom.push_back(&pair_data_b);
  contrastive_loss_bottom.push_back(&pair_label);
  
  contrastive_loss_top.clear();
  contrastive_loss_top.push_back(top[0]);

  contrastive_loss_layer_->SetUp(contrastive_loss_bottom, contrastive_loss_top);
  
  contrastive_propogate.clear();
  contrastive_propogate.push_back(true);
  contrastive_propogate.push_back(true);
  contrastive_propogate.push_back(false);  
  
}


template <typename Dtype>
void DistanceWeightedValidPairLossV12Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  //set the input data
  for (int i = 0; i < pair_num; i++){
    //The pair label
    if( bottom[1]->cpu_data()[pair_index_a[i]] == bottom[1]->cpu_data()[pair_index_b[i]] ){
      pair_label.mutable_cpu_data()[i] = Dtype(1);
    }
    else{
      pair_label.mutable_cpu_data()[i] = Dtype(0);
    }
    //the pair data
    const Dtype* source_a = bottom[0]->cpu_data() + pair_index_a[i] * feature_dim;
    const Dtype* source_b = bottom[0]->cpu_data() + pair_index_b[i] * feature_dim;
    Dtype* dest_a = pair_data_a.mutable_cpu_data() + i*feature_dim;
    Dtype* dest_b = pair_data_b.mutable_cpu_data() + i*feature_dim;

    caffe_copy(feature_dim, source_a, dest_a);
    caffe_copy(feature_dim, source_b, dest_b);
  }
  // The forward pass of contrastive loss.
  contrastive_loss_layer_->Forward(contrastive_loss_bottom, contrastive_loss_top);
}

template <typename Dtype>
void DistanceWeightedValidPairLossV12Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    // The backward pass of contrastive loss.
    contrastive_loss_layer_->Backward(contrastive_loss_top, contrastive_propogate, contrastive_loss_bottom);
    // set diff to zero
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
    //dynamic cast
    shared_ptr<ContrastiveLossV2Layer<Dtype> > auxiLayer = boost::dynamic_pointer_cast<ContrastiveLossV2Layer<Dtype> >(contrastive_loss_layer_);
    if (!auxiLayer) {
      throw std::runtime_error("ContrastiveLossV2Layer dynamic pointer_cast failed");
    }

    //reweighting
    Dtype margin = this->layer_param_.contrastive_loss_param_v2().margin();
    Dtype pn_margin = this->layer_param_.contrastive_loss_param_v2().pn_margin();
    //The positive parameters of power_x_scale_base: temporature parameter (T=scale)
    Dtype scale = this->layer_param_.softmax_t_p_param().scale();
    Dtype base = this->layer_param_.softmax_t_p_param().base();
    //The negative parameters of power_x_scale_base:
    Dtype scale_n = this->layer_param_.softmax_t_n_param().scale();
    Dtype base_n = this->layer_param_.softmax_t_n_param().base();
    
    vector<Dtype> pair_weights(pair_num, Dtype(0));
    vector<Dtype> sum_sample_weights_sim(sample_num, static_cast<Dtype>(1e-8));
    vector<Dtype> sum_sample_weights_dissim(sample_num, static_cast<Dtype>(1e-8));
    Dtype sum_pair_weight = static_cast<Dtype>(1e-8);

    
    for(int i = 0; i < pair_num; i++){
      Dtype dist = sqrt(auxiLayer->dist_sq_.cpu_data()[i]);
      if( static_cast<int>(pair_label.cpu_data()[i]) ){
        //similar pairs
        //transformation to weight
        Dtype m_dist = dist - (margin - pn_margin);

        //weight computation
        if( m_dist > Dtype(0.0) ){
          pair_weights[i] = power_x_scale_base(m_dist, scale, base);
        }
        else{
          pair_weights[i] = Dtype(0.0);
        }
        

        sum_sample_weights_sim[pair_index_a[i]] += pair_weights[i];
        sum_sample_weights_sim[pair_index_b[i]] += pair_weights[i];
      }
      else{
        //dissimilar pairs
        Dtype m_dist = margin - dist;
        if( m_dist > Dtype(0.0) ){
          pair_weights[i] = power_x_scale_base(m_dist, scale_n, base_n);
        }
        else{
          pair_weights[i] = Dtype(0.0);
        }
        

        sum_sample_weights_dissim[pair_index_a[i]] += pair_weights[i];
        sum_sample_weights_dissim[pair_index_b[i]] += pair_weights[i];
      }
      //sum_pair_weight += pair_weights[i];
    }
    for(int i = 0; i < sample_num; i++){
      sum_pair_weight += sum_sample_weights_sim[i];
      sum_pair_weight += sum_sample_weights_dissim[i];
    }

    //copy gradients and scale the gradients with normalized score
    for(int i = 0; i < pair_num; i++){
      
      Dtype weight_a = Dtype(0);
      Dtype weight_b = Dtype(0);
      if( static_cast<int>(pair_label.cpu_data()[i]) ){
        //similar pairs
        weight_a = 0.5 * pair_weights[i] / sum_sample_weights_sim[pair_index_a[i]];
        weight_b = 0.5 * pair_weights[i] / sum_sample_weights_sim[pair_index_b[i]];
      }
      else{
        //dissimilar pairs
        weight_a = 0.5 * pair_weights[i] / sum_sample_weights_dissim[pair_index_a[i]];
        weight_b = 0.5 * pair_weights[i] / sum_sample_weights_dissim[pair_index_b[i]];
      }
      weight_a *= (sum_sample_weights_sim[pair_index_a[i]] + sum_sample_weights_dissim[pair_index_a[i]]) / sum_pair_weight;
      weight_b *= (sum_sample_weights_sim[pair_index_b[i]] + sum_sample_weights_dissim[pair_index_b[i]]) / sum_pair_weight;
      
      const Dtype* source_a = pair_data_a.cpu_diff() + i * feature_dim;
      const Dtype* source_b = pair_data_b.cpu_diff() + i * feature_dim;
      Dtype* dest_a = bottom[0]->mutable_cpu_diff() + pair_index_a[i] * feature_dim;
      Dtype* dest_b = bottom[0]->mutable_cpu_diff() + pair_index_b[i] * feature_dim;
      //dest[ind] = weight * source[ind] + dest[ind]
      caffe_axpy(feature_dim, weight_a, source_a, dest_a);
      caffe_axpy(feature_dim, weight_b, source_b, dest_b);
    }
  }
  if (propagate_down[1] || propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label and score inputs.";
  }

}

#ifdef CPU_ONLY
STUB_GPU(DistanceWeightedValidPairLossV12Layer);
#endif

INSTANTIATE_CLASS(DistanceWeightedValidPairLossV12Layer);
REGISTER_LAYER_CLASS(DistanceWeightedValidPairLossV12);

}  // namespace caffe
