%%
%pause(1*60*60*4.0)

addpath('../pre_pro_process');
addpath('../pre_pro_process/utils');
load( 'InshopClothes_EvaImagePathBoxCell.mat' );
addpath ../../CaffeMex_RLL_GR_V03_Simp/matlab/
mainDir = '../';
modelDir = 'deploy_prototxts';



%%%%%%%%%%%Configurations Start%%%%%%%
param.gpu_id = 3;
param.test_batch_size = 128;
param.test_net_file = fullfile(mainDir, modelDir, 'test_H_d128.prototxt');
blob_name = 'pool5/7x7_s1';
record_file = 'Recall_H_d128.txt';
%
param.save_start = 2000;
param.save_interval = 2000;
param.train_maxiter = 30000;
%%%%%%%%%%%Configurations End%%%%%%%




param.save_model_file = 'checkpoints';
param.save_model_name = 'checkpoint_iter';
trial_index = 1;
param.use_gpu = 1;
gpuDevice(param.gpu_id + 1);
param.force_square_size = 256;
param.input_size = 227;
param.cropped_scale = [0.16,1];
param.aspect_ratio = [3/4, 4/3];
param.flip_prob = 0.5;

for iter = param.save_start : param.save_interval : param.train_maxiter
        cur_path = pwd;
        caffe.reset_all;
        caffe.set_mode_gpu();
        caffe.init_log(fullfile(cur_path, 'log'));

        model_path = strcat(param.save_model_file, num2str(trial_index),...
                                                '/', param.save_model_name, '_', num2str(iter), '.caffemodel');
        net = caffe.get_net(param.test_net_file, model_path, 'test');

        

        test_ori_center_crop_v04_InshopClothes;
end

exit;
