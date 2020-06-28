
 %% configs
 % default
 force_square_size = 256;
 cropped_scale = [0.16,1];
 aspect_ratio = [3/4, 4/3];
 input_size = 227;
 istrain = true;
 flip_prob = 0.5;
 %%     
  
 %img_name = 'Black_Footed_Albatross_0001_796111.jpg';
 img_name = 'Black_Footed_Albatross_0014_89.jpg';

 for i = 1 : 1000
     process_ori_image_aug_v04_21042020(img_name, ... 
        force_square_size, input_size, ...
        cropped_scale, aspect_ratio, ...
        flip_prob, istrain);
 end
