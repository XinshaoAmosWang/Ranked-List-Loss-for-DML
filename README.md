# Ranked-List-Loss-for-Deep-Metric-Learning

### [[Paper]](https://arxiv.org/abs/1903.03238) [[Slides]](https://drive.google.com/file/d/1nSXCe-7t_EkNwjFuXTnmzzoFr-6jFKVW/view?usp=sharing) [[Poster]](https://drive.google.com/file/d/1vSp3mDRJKdQFNUH12ehuDDyqQfjXFnWM/view?usp=sharing)


## Dependencies
The core functions are implemented in the [caffe](https://github.com/BVLC/caffe) framework. We use matlab interfaces matcaffe for data preparation.  
* [CaffeMex_v2](https://github.com/sciencefans/CaffeMex_v2/tree/9bab8d2aaa2dbc448fd7123c98d225c680b066e4)
* MATLAB 2017b


## Setup

Clone the submodule and our repository:

```bash
git clone git@github.com:XinshaoAmosWang/Ranked-List-Loss-for-Deep-Metric-Learning.git
```

Put the files of new layers to the corresponding directories of submodule
[CaffeMex_v2](https://github.com/sciencefans/CaffeMex_v2/tree/9bab8d2aaa2dbc448fd7123c98d225c680b066e4)

## Usage

Examples for reproducing our results on [Stanford Online Product dataset](http://cvgl.stanford.edu/projects/lifted_struct/) are given. 

### Citation
If you find our code and paper useful in your research, please kindly cite our paper:

	@inproceedings{wang2019ranked,
        title={Ranked List Loss for Deep Metric Learning},
        author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Hu, Guosheng and Garnier, Romain and Robertson, Neil M},
        journal=cvpr,
        year={2019}
    }


## More Qualitative results
[[Slides]](https://drive.google.com/file/d/1nSXCe-7t_EkNwjFuXTnmzzoFr-6jFKVW/view?usp=sharing) [[Poster]](https://drive.google.com/file/d/1vSp3mDRJKdQFNUH12ehuDDyqQfjXFnWM/view?usp=sharing)
## Common questions

#### 1. What does ranking mean? 
The overall objective is make the postive set ranks before the negative set by a distance margin. We do not consider the exact order of examples within the positive set and negative set. 

#### 2. What are the key components which influence the performance a lot?
* Sample mining
* Sample weighting
* Two distance hyper-parameters for optimisation and regularisation jointly.
* Exploiting a weighted combination of more data points.  

