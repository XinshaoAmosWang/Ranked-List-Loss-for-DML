function [batch_data, batch_label] = get_train_minibatch_ori_aug_v02 ( id_num, image_per_id, ...
                                        batch_data, batch_label, batch_size, ...
                                        class_ids,  TrainImagePathBoxCell, ...
                                        crop_padding, force_square_size)
    % sample severals ids
    ids = unique(class_ids);
    select_id_indexes = randperm(length(ids), id_num);
    
    for i = 1 : id_num
        % one id
        select_id = ids(select_id_indexes(i));
        % all the images of this id
        indexes = ( class_ids == select_id );
        images_of_this_id = TrainImagePathBoxCell(indexes, :);
        %
        % the images of this ID
        len = length(images_of_this_id);
        if(len >= image_per_id)
            sample_index = randperm(len, image_per_id);
        else
            sample_index = mod(randperm(image_per_id), len) + 1;
        end

        sampled_images_of_this_id = images_of_this_id(sample_index, :);
        %
        %
        for img = 1 : image_per_id
            batch_label(:,:,:, (i-1)*image_per_id + img ) = select_id - 1;  %minus one -> [0, class-num-1]
            batch_data(:,:,:, (i-1)*image_per_id + img ) = process_ori_image_aug_v02(sampled_images_of_this_id{img,1}, ...
                                                                                    crop_padding, force_square_size, ...
                                                                                    true);
        end

    end
end