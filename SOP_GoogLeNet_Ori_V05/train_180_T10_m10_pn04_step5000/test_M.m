%%
%pause(1*60*60*4.0)

addpath('../pre_pro_process');
addpath('../pre_pro_process/utils');

addpath ../../CaffeMex_v2/matlab/
mainDir = '../';

modelDir = '180_T10_m12_pn04_step5000';
param.gpu_id = x;
param.test_batch_size = 64;
param.test_net_file = fullfile(mainDir, modelDir, 'test_M.prototxt');


param.save_model_file = 'SOP';
param.save_model_name = 'SOP_iter';
load( 'SOP_TestImagePathBoxCell.mat' );
trial_index = 1;
param.use_gpu = 1;
gpuDevice(param.gpu_id + 1);
param.crop_padding = 15;
param.force_square_size = 256;

for iter = 10000 : 2000 : 20000
        cur_path = pwd;
        caffe.reset_all;
        caffe.set_mode_gpu();
        caffe.set_device(param.gpu_id);
        caffe.init_log(fullfile(cur_path, 'log'));

        model_path = strcat(param.save_model_file, num2str(trial_index),...
                                                '/', param.save_model_name, '_', num2str(iter), '.caffemodel');
        net = caffe.get_net(param.test_net_file, model_path, 'test');

        record_file = 'Recall.txt';
        test_ori_center_crop_v02;
end

exit;
