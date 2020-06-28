function [img] = process_ori_image_aug_v04_21042020(img_name, ... 
        force_square_size, input_size, ...
        cropped_scale, aspect_ratio, ...
        flip_prob, istrain)
    % Read image and pre-process
    % 1. resize to (force_square_size,force_square_size)
    % 2. random crop or center crop => input size
    % 3. permute channels
    % 4. subtract mean value for each channel

    % Read the RGB image(orig_img) 
    orig_img = imread( img_name );
    if (ndims(orig_img) >=3)
        %view = rgb2gray(orig_img);
    else
        % if read as gray scale, convert orig image to rgb
        view = orig_img;
        orig_img = cat(3, view, view, view);
    end

    % Resize to force_square_size
    [height, width, dd] = size(orig_img);
    orig_img = single(orig_img);
    orig_img = imresize(orig_img, [force_square_size force_square_size]);
    
    % Flip and Crop: orig_img is of size(force_square_size, force_square_size)
    img = orig_img; % default: 256x256
    if istrain
        original_ratio =  1; % after resize
        random_ratio = rand(1)*(aspect_ratio(2)-aspect_ratio(1)) + aspect_ratio(1);
        crop_ratio = original_ratio * random_ratio;
        scale = rand(1)*(cropped_scale(2)-cropped_scale(1)) + cropped_scale(1);

        new_height = floor(force_square_size * scale * crop_ratio);
        new_width = floor(force_square_size * scale); 
        if new_height >= force_square_size
            new_height = force_square_size-1;
            new_width = floor(new_height/crop_ratio); % keep the ratio
        end
        if new_width >= force_square_size
            new_width = force_square_size-1;
            new_height = floor(new_width * crop_ratio); % keep the ratio
        end

        if( rand(1) < 0.5 )
            img = flip(img, 2);
        end
        img = img_random_crop(img, new_height, new_width);
        img = imresize(img, [input_size input_size]);
    else
        img = img_center_crop(img, input_size, input_size);
    end
    
    img = img(:,:,[3,2,1]); %Convert RGB to BGR
    img = permute(img,[2,1,3]); %Switch width and height
    
    img(:,:,1) = img(:,:,1) - 104;
    img(:,:,2) = img(:,:,2) - 117;
    img(:,:,3) = img(:,:,3) - 123;


end

