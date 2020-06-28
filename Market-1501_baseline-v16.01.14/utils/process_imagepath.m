function [img] = process_imagepath(img_path, crop_padding, force_square_size, cropped_size, istrain)
	%Read image and pre-process
	%   1. resize to (force_square_size,force_square_size)
	%   2. permute channels
	%   3. subtract mean value for each channel

	% Read the RGB image(orig_img) and gray scale image(view)
	orig_img = imread( img_path );
	if (ndims(orig_img) >=3)
        view = rgb2gray(orig_img);
    else
        % if read as gray scale, convert orig image to rgb
        view = orig_img;
        orig_img = cat(3, view, view, view);
    end

    % General Crop and Rescale with fixed aspect ratio
    
    [row, col] = find(view < 250);
    %[row, col, ~] = size(view); %this is a bug. 

    xmin = max(0, min(col) - crop_padding);
    xmax = min(size(view, 2), max(col) + crop_padding);
    ymin = max(0, min(row) - crop_padding);
    ymax = min(size(view, 1), max(row) + crop_padding);

    width = xmax - xmin + 1;
    height = ymax - ymin + 1;
    img_cropped = imcrop(orig_img, [xmin, ymin, width, height]);

    % hos: recompute width, height (sometimes gets cropped out of bound)
    [height, width, dd] = size(img_cropped);
    img_cropped = single(img_cropped);
    
    if force_square_size > 0
        if height > width
            img_cropped = imresize(img_cropped, [force_square_size NaN]);
            pre_padding = floor((force_square_size - size(img_cropped, 2)) / 2);
            img_cropped = padarray(img_cropped, [0 pre_padding 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 2);
            img_cropped = padarray(img_cropped, [0 post_padding 0], 255, 'post');
        elseif width > height
            img_cropped = imresize(img_cropped, [NaN force_square_size]);
            pre_padding = floor((force_square_size - size(img_cropped, 1)) / 2);
            img_cropped = padarray(img_cropped, [pre_padding 0 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 1);
            img_cropped = padarray(img_cropped, [post_padding 0 0], 255, 'post');
        else
            img_cropped = imresize(...
                img_cropped, [force_square_size force_square_size]);
        end
        assert(size(img_cropped, 1) == ...
            force_square_size && size(img_cropped, 2) == force_square_size);
    end





    % Flip and Crop
    img = img_cropped;
    crop_h = cropped_size;
    crop_w = cropped_size;
    if istrain
        if( rand(1) < 0.5 )
            img = flip(img, 2);
        end
        img = img_random_crop(img, crop_h, crop_w);
    else
        img = img_center_crop(img, crop_h, crop_w);
    end

    img = img(:,:,[3,2,1]); %Convert RGB to BGR
    img = permute(img,[2,1,3]); %Switch width and height
    img(:,:,1) = img(:,:,1) - 104;
    img(:,:,2) = img(:,:,2) - 117;
    img(:,:,3) = img(:,:,3) - 123;
end

