function [features_output] = process_imagescell_net_blobname(net, imagescell, batch_size, ...
                                            crop_padding, force_square_size,  cropped_size, blobname)
% Summary of this function goes here
%The fixed quality version
%   1. Input
%       net: the trained model
%       imagescell: image names cell
%       batch_size: the maximum input for each forward
%   2. Process
%       a. form batch
%       b. forward
%       c. extract output features
%   3. Output: (length(imagescell), dim)
features_output = [];

num_img = length(imagescell);
num_batch = floor(num_img/batch_size);

net.blobs('data').reshape([cropped_size cropped_size 3 batch_size]);

for bt = 1 : num_batch
    % form batch
    batch_data = zeros(cropped_size, cropped_size, 3, batch_size, 'single');
    for ind = 1 : batch_size
        ind_img = (bt-1)*batch_size + ind;
        img_path = imagescell{ind_img};
        batch_data(:,:,:, ind) = process_imagepath(img_path, ...
                                    crop_padding, force_square_size, cropped_size, ...
                                    false);
    end
    % set_data, forward
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    % extract features
    features_output = [features_output; (squeeze(net.blobs(blobname).get_data))'];
end

% remained images
if  mod(num_img, batch_size) ~= 0
    remained_images = imagescell(num_batch*batch_size + 1 : end);
    num_img_remained = length(remained_images);
    batch_data = zeros(cropped_size, cropped_size, 3, num_img_remained, 'single');

    net.blobs('data').reshape([cropped_size cropped_size 3 num_img_remained]);

    for ind = 1 : num_img_remained
        img_path = remained_images{ind};
        batch_data(:,:,:, ind) = process_imagepath(img_path, ...
                                    crop_padding, force_square_size, cropped_size, ...
                                    false);
    end

    % set data, forward
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    % extract features
    features_output = [features_output; (squeeze(net.blobs(blobname).get_data))'];
end

end

