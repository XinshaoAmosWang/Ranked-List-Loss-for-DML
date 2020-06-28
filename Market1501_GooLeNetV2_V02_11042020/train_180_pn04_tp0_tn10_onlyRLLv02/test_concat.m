%%
%pause(1*60*60*4.0)

addpath('../pre_pro_process');
addpath('../pre_pro_process/utils');
load( 'Market1501_evaluation.mat' );
addpath ../../CaffeMex_RLL_GR_V03_Simp/matlab/
mainDir = '../';
modelDir = 'deploy_prototxts';



%%%%%%%%%%%Configurations Start%%%%%%%
param.gpu_id = 2;
param.test_batch_size = 128;
param.test_net_file = fullfile(mainDir, modelDir, 'test_concat.prototxt');
blob_name = 'pool5/7x7_s1';
record_file = 'Record_concat.txt';
%
param.save_start = 3000;
param.save_interval = 1000;
param.train_maxiter = 6000;
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

        

        test_Market1501_v02;
end

exit;
