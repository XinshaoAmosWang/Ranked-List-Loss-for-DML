function [img] = process_box_image_aug_v01(img_name, box, istrain)
%Read image and pre-process
%   1. resize to (224,224)
%   2. permute channels
%   3. subtract mean value for each channel
img = imread( img_name );
x = box(1);
y = box(2);
width = box(3);
height = box(4);

% crop by bounding boxes
img = img(max(y,1) : min(max(y,1) + height-1, size(img,1)), ...
	      max(x,1) : min(max(x,1) + width-1, size(img,2)), ...
	      :);

% Resize to target_wd * target_ht.
img = imresize(single(img), [256 256], 'bicubic');

% Normalization
img = img / 255.0;
%process for single channel
if size(img,3) == 1
    img = cat(3, img, img, img);
end
img(:,:,1) = (img(:,:,1) - 0.485) / 0.229;
img(:,:,2) = (img(:,:,2) - 0.456) / 0.224;
img(:,:,3) = (img(:,:,3) - 0.406) / 0.225;


% Flip and Crop
crop_h = 224;
crop_w = 224;
if istrain
	if( rand(1) < 0.5 )
		img = flip(img, 2);
	end
	img = img_random_crop(img, crop_h, crop_w);
else
	img = img_center_crop(img, crop_h, crop_w);
end

img = img(:,:,[3,2,1]); %%??
img = permute(img,[2,1,3]); %%??
        

end

