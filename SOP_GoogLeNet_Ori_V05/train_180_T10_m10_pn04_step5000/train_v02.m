%pause(1*60*60*5)

seed = 123;
rng(seed);
%% train network config
addpath('../pre_pro_process');
addpath('../pre_pro_process/utils');
load('SOP_TrainImagePathBoxCell.mat')
addpath ../../CaffeMex_v2/matlab/
mainDir = '../';



param.gpu_id = [2];
modelDir = '180_T10_m10_pn04_step5000';
param.fintune_model = fullfile(mainDir, 'pretrain_model', 'googlenet_bn.caffemodel');
id_num = 60;
image_per_id = 3;





param.solver_netfile = fullfile(mainDir, modelDir, 'solver.prototxt');
param.save_model_file = 'SOP';
param.save_model_name = 'SOP_iter';
%
param.save_start = 2000;
param.save_interval = 2000;
param.train_maxiter = 20000;
param.output_interval = 1000;
%
param.use_gpu = 1;
gpuDevice([]);
for g = 1 : length(param.gpu_id);
    gpuDevice(param.gpu_id(g) + 1);
end
param.crop_padding = 15;
param.force_square_size = 256;
%%
trial_index = 1;

    if ~exist(strcat(param.save_model_file, num2str(trial_index)),'file')
        mkdir(strcat(param.save_model_file, num2str(trial_index)));
    end

    %% find caffe -> reset_all ->init_log -> set_mode_gpu
    cur_path = pwd;
    caffe.reset_all;

    %caffe fix seed
    if param.use_gpu
        caffe.set_mode_gpu;
        caffe.set_random_seed(seed);
        %caffe.set_device(param.gpu_id);
        %for g = 1 : length(param.gpu_id);
                %    caffe.set_device(param.gpu_id(g) + 1);
                %end
    else
        caffe.set_mode_cpu;
    end
    caffe.init_log(fullfile(cur_path,'log'));

    %% caffe model init
    caffe_solver = caffe.get_solver(param.solver_netfile, param.gpu_id);
    caffe_solver.use_caffemodel(param.fintune_model);

    %% Train the model
    % batch data for each iteration
    input_data_shape = caffe_solver.nets{1}.blobs('data').shape;
    batch_size = input_data_shape(4);


    %% data preparation for multiple GPUs
    gpu_num = length(param.gpu_id);
    id_num = id_num * gpu_num;
    batch_size = batch_size * gpu_num;
    input_data_shape(4) = batch_size;


    batch_data = zeros(input_data_shape, 'single');
    batch_label = zeros(1,1,1,batch_size, 'single');
    %
    assert( batch_size == id_num * image_per_id );
    image_attention = ones(1,1,1, batch_size, 'single') / batch_size;  %% something wrong there, but okay since we do not use attention
    %
    train_x_axis=[];
    train_y_axis=[];




    ids = unique(class_ids);
    class_num = length(ids);
    valid_num = floor(class_num/id_num)*id_num;% make mode = 0
    select_id_indexes = randperm(class_num, valid_num);

    iter = 1;
    epoch = 1;
    base = 0;
    while iter < param.train_maxiter
        if ( iter * id_num  > epoch * valid_num ) % reach the end of epoch
            epoch = epoch + 1;
            select_id_indexes = randperm(class_num, valid_num); % make mode = 0
            base = base + valid_num;
        end

        %% get and set batch data
        [batch_data, batch_label] = get_train_minibatch_ori_aug_v03 ( id_num, image_per_id, ...
                                        batch_data, batch_label, batch_size, ...
                                        class_ids,  TrainImagePathBoxCell, ...
                                        param.crop_padding, param.force_square_size, ...
                                        ids, select_id_indexes( (iter-1)*id_num+1-base : iter*id_num-base ) );


        perGPU = id_num / gpu_num;
        perGPU = perGPU * image_per_id;
        for g = 1 : gpu_num
            caffe_solver.nets{g}.blobs('data').set_data( batch_data(:,:,:, (g-1)*perGPU+1 : g*perGPU) );
            caffe_solver.nets{g}.blobs('label').set_data( batch_label(:,:,:, (g-1)*perGPU+1 : g*perGPU) );
            % attention input
            caffe_solver.nets{g}.blobs('image_attention').set_data(image_attention(:,:,:, (g-1)*perGPU+1 : g*perGPU));
        end


        %
        % launch one step of gradient descent
        caffe_solver.step(1);

        %% print && plot loss -> drawnow
        iter = caffe_solver.iter;
        if mod(iter, param.output_interval) == 0

            WSiamese_loss_image = caffe_solver.nets{1}.blobs('WSiamese/loss_image').get_data;
            WSiamese_loss_image1 = caffe_solver.nets{1}.blobs('WSiamese/loss_image1').get_data;
            WSiamese_loss_image2 = caffe_solver.nets{1}.blobs('WSiamese/loss_image2').get_data;

            train_x_axis = [train_x_axis, iter];
            train_y_axis = [train_y_axis, WSiamese_loss_image];
            plot(train_x_axis, train_y_axis);
            drawnow;
            fprintf('epoch= %d, iter= %d, WSiamese_loss_image=%f, WSiamese_loss_image1=%f, WSiamese_loss_image2=%f\n',...
                epoch, iter, WSiamese_loss_image, WSiamese_loss_image1, WSiamese_loss_image2);

        end

        %% save model 
        if iter >= param.save_start && mod(iter, param.save_interval) == 0
            %save
            model_name = strcat(param.save_model_file,num2str(trial_index),...
                                    '/',param.save_model_name,'_',num2str(iter));
            caffe_solver.nets{1}.save(strcat(model_name, '.caffemodel'));

            % save solverstate
            caffe_solver.savestate( strcat(model_name, '_snapshot_') );

            mat_path = strcat('error', num2str(iter), '.mat');
            save(mat_path, 'train_x_axis', 'train_y_axis');
        end
    end

    mat_path = strcat('error', num2str(iter), '.mat');
    save(mat_path, 'train_x_axis', 'train_y_axis');




exit;
