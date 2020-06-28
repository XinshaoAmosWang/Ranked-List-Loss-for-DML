function [img] = img_center_crop(img, crop_h, crop_w)
% crop_h, crop_w: The height and width to crop.   

% cut out
height = size(img, 1);
width = size(img, 2);

h_start = ceil( 0.5 * (height - crop_h) );
w_start = ceil( 0.5 * (width - crop_w) );
h_end = h_start + crop_h - 1;
w_end = w_start + crop_w - 1;

img = img(h_start:h_end, w_start:w_end, :);

end

