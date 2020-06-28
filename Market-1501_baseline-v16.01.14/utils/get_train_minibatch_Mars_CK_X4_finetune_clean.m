function [batch_data, batch_label, tracklet_label] = get_train_minibatch_Mars_CK_X4_finetune_clean (tracklet_num, frame_per_tracklet, ...
                                        batch_data, batch_label, batch_size, ...
                                        tracklets, labels, cleaned_labels, ...
                                        crop_padding, force_square_size,  cropped_size)
    % sample severals ids
    ids = unique(labels);
    selects = randperm(length(ids), tracklet_num/4);
    selects = [selects selects selects selects];
    
    for i = 1 : tracklet_num
        % one id
        select_id = ids(selects(i));
        % one tracklet of this id
        indexes = ( labels == select_id );
        tracklets_of_this_id = tracklets(indexes);
        labels_of_this_id = cleaned_labels(indexes);
        one_ind = randperm( length(tracklets_of_this_id), 1);
        select_tracklet = tracklets_of_this_id{one_ind};
        select_label = labels_of_this_id{one_ind};
        %
        % the images in this tracklet
        len = length(select_tracklet);
        if(len >= frame_per_tracklet)
            sample_index = randperm(len, frame_per_tracklet);
        else
            sample_index = mod(randperm(frame_per_tracklet), len) + 1;
        end

        sub_tracklet = select_tracklet(sample_index);
        sub_label = select_label(sample_index);
        %
        %
        tracklet_label(:,:,:, i) = select_id;
        for fr = 1 : frame_per_tracklet
            batch_label(:,:,:, (i-1)*frame_per_tracklet + fr ) = sub_label(fr);
            img_path = sub_tracklet{fr};
            batch_data(:,:,:, (i-1)*frame_per_tracklet + fr ) = process_imagepath(img_path, ...
                                                    crop_padding, force_square_size, cropped_size, ...
                                                    true);
        end

    end
end
