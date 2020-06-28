
root = '/home/xinshao/Papers_Projects/Data/InshopClothesRetrievalBenchmark';

[image_names,  image_id_str,  image_types_str ] =  ...
    textread('list_eval_partition.txt', '%s %s %s', ...
    'headerlines', 2);

% ids , types convert
image_ids = zeros( length(image_id_str), 1 );
image_types = zeros( length(image_id_str), 1 );
for i = 1 : length(image_id_str)
    image_ids(i) = str2double( image_id_str{i}(4:end) );
    % type
    if strcmp(image_types_str{i} , 'train')
        image_types(i) = 0;
    elseif strcmp(image_types_str{i} , 'query')
        image_types(i) = 1;
    else
        image_types(i) = 2;
    end
    % image name 
    image_names{i} = fullfile(root, image_names{i});
end

% id relabelling
image_relabel_ids = image_ids;
unique_ids = unique(image_ids);
for id_index  = 1 : length(unique_ids)
    positions = image_ids == unique_ids(id_index);
    % change the labelling
    image_relabel_ids(positions) = id_index;
end


% Image types 
train_indexes = (image_types == 0);
query_indexes = (image_types == 1);
gallery_indexes = (image_types == 2);
%=>
train_names = image_names(train_indexes);
train_ids = image_relabel_ids(train_indexes);
query_names = image_names(query_indexes);
query_ids = image_relabel_ids(query_indexes);
gallery_names = image_names(gallery_indexes);
gallery_ids = image_relabel_ids(gallery_indexes);


%% Highlight
%% To correct the information about this dataset
% Total: images num: 52712, class_num: 7982
% Training: images_num: 25,882, class_num: 3997
% Query: images_num: 14,218, class_num: 3,985 
% Gallery: images_num: 12,612, class_num: 3,985 
% Overlap between training & evaluation?: 
% sum(unique(query_ids)==unique(gallery_ids)) = 3985
%
% k = -[1 : 1 : (3997-3985)]'
% sum(unique([query_ids; k])==unique(train_ids)) = 0, not overlapped


save_path = 'InshopClothes_TrainImagePathBoxCell.mat';
TrainImagePathBoxCell = train_names;
class_ids = train_ids;
% id relabelling
class_ids = train_ids;
unique_ids = unique(train_ids);
for id_index  = 1 : length(unique_ids)
    positions = train_ids == unique_ids(id_index);
    % change the labelling
    class_ids(positions) = id_index;
end
save(save_path,...
        'TrainImagePathBoxCell', ...
        'class_ids', ...
        '-v7.3');
 
save_path = 'InshopClothes_EvaImagePathBoxCell.mat';
TestImagePathBoxCell = [query_names; gallery_names];
test_ids = [query_ids; gallery_ids];
% id relabelling
class_ids = test_ids;
unique_ids = unique(test_ids);
for id_index  = 1 : length(unique_ids)
    positions = test_ids == unique_ids(id_index);
    % change the labelling
    class_ids(positions) = id_index;
end
save(save_path,...
        'EvaImagePathBoxCell', ...
        'class_ids', ...
        '-v7.3');
