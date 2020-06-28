function [images_feature] = forward_ori_center_crop_v04(net, TestImagePathBoxCell, ...
                                            batch_size, ...
                                            force_square_size, input_size,...
                                            cropped_scale, aspect_ratio, ...
                                            flip_prob, ...
                                            blob_name)
%forward_images_data 
%Summary of this function goes here
%   1. Input
%       net: the trained model
%       TestImagePathBoxCell: image data cell
%       batch_size: the maximum input for each forward
%   2. Process
%       a. form batch
%       b. forward
%       c. extract output features
%   3. Output: (n, dim)
images_feature = [];

num_img = size(TestImagePathBoxCell, 1);
num_batch = floor(num_img/batch_size);

net.blobs('data').reshape([input_size input_size 3 batch_size]);

for bt = 1 : num_batch
    % form batch
    batch_data = zeros(input_size, input_size, 3, batch_size, 'single');
    for ind = 1 : batch_size
        ind_img = (bt-1)*batch_size + ind;
        batch_data(:,:,:, ind) = ...
                    process_ori_image_aug_v04_21042020(TestImagePathBoxCell{ind_img, 1}, ...
                        force_square_size, input_size, ...
                        cropped_scale, aspect_ratio, ...
                        flip_prob, false);
    end
    % set_data, forward
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    % extract features
    images_feature = [images_feature; (squeeze(net.blobs(blob_name).get_data))'];
end

% remained images
if  mod(num_img, batch_size) ~= 0
    remained_images = TestImagePathBoxCell(num_batch*batch_size + 1 : end, :);
    num_img_remained = size(remained_images, 1);

    batch_data = zeros(input_size, input_size, 3, num_img_remained, 'single');
    net.blobs('data').reshape([input_size input_size 3 num_img_remained]);

    for ind = 1 : num_img_remained
        batch_data(:,:,:, ind) = ...
                    process_ori_image_aug_v04_21042020(remained_images{ind, 1}, ...
                        force_square_size, input_size, ...
                        cropped_scale, aspect_ratio, ...
                        flip_prob, false);
    end

    % set data, forward
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    % extract features
    images_feature = [images_feature; (squeeze(net.blobs(blob_name).get_data))'];
end


end

