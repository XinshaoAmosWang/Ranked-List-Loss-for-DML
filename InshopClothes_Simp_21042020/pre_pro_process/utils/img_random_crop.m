function [img] = img_random_crop(img, crop_h, crop_w)
% crop_h, crop_w: The height and width to crop.   
% crop randomly

% cut out
height = size(img, 1);
width = size(img, 2);

h_start = ceil( rand(1) * (height - crop_h) );
w_start = ceil( rand(1) * (width - crop_w) );
h_end = h_start + crop_h - 1;
w_end = w_start + crop_w - 1;

img = img(h_start:h_end, w_start:w_end, :);

end

