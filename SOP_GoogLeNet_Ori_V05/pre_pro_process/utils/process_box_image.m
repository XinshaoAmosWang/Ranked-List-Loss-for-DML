function [im_data] = process_box_image(img_name, x, y, width, height)
%Read image and pre-process
%   1. resize to (224,224)
%   2. permute channels
%   3. subtract mean value for each channel
img = imread( img_name );

% crop by bounding boxes
img = img(max(y,1) : min(max(y,1) + height-1, size(img,1)), ...
	      max(x,1) : min(max(x,1) + width-1, size(img,2)), ...
	      :);

%process for single channel
if size(img,3) == 1
    img = cat(3, img, img, img);
end

im_data = imresize((single(img)),[224 224]);
im_data = im_data(:,:,[3,2,1]);
im_data = permute(im_data,[2,1,3]);
        
im_data(:,:,1) = im_data(:,:,1)-104;
im_data(:,:,2) = im_data(:,:,2)-117;
im_data(:,:,3) = im_data(:,:,3)-123;

end

