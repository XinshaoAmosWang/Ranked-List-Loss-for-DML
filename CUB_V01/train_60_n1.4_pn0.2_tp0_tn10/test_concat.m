%%
%pause(1*60*60*4.0)

addpath('../pre_pro_process');
addpath('../pre_pro_process/utils');
load( 'CUB_200_2011_TestImagePathBoxCell.mat' );
TestImagePathBoxCell = CUB_200_2011_TestImagePathBoxCell;
clear CUB_200_2011_TestImagePathBoxCell;
addpath ../../CaffeMex_v2/matlab/
mainDir = '../';
modelDir = 'deploy_prototxts';



%%%%%%%%%%%Configurations Start%%%%%%%
param.gpu_id = 1;
param.test_batch_size = 64;
param.test_net_file = fullfile(mainDir, modelDir, 'test_concat.prototxt');
blob_name = 'pool5/7x7_s1';
record_file = 'Recall_concat.txt';
%
param.save_start = 500;
param.save_interval = 500;
param.train_maxiter = 10000;
%%%%%%%%%%%Configurations End%%%%%%%




param.save_model_file = 'checkpoints';
param.save_model_name = 'checkpoint_iter';
trial_index = 1;
param.use_gpu = 1;
gpuDevice(param.gpu_id + 1);
param.crop_padding = 15;
param.force_square_size = 256;
param.cropped_size = 227;

for iter = param.save_start : param.save_interval : param.train_maxiter
        cur_path = pwd;
        caffe.reset_all;
        caffe.set_mode_gpu();
        caffe.init_log(fullfile(cur_path, 'log'));

        model_path = strcat(param.save_model_file, num2str(trial_index),...
                                                '/', param.save_model_name, '_', num2str(iter), '.caffemodel');
        net = caffe.get_net(param.test_net_file, model_path, 'test');

        

        test_ori_center_crop_v03;
end

exit;
