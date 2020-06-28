
%% add necessary paths
root = '/home/xinshao/Papers_Projects/Data/Market-1501-v15.09.15';
train_dir = fullfile(root, 'bounding_box_train/');% train directory
save_path = 'Market1501TrainImagePathBoxCell.mat';
%pre ='';
%% calculate the ID and camera for database images
files = dir([train_dir '*.jpg']);
IDs = zeros(length(files), 1);
CAMs = zeros(length(files), 1); % not used during training
ImagePathBoxCell  = cell(length(files), 1);

    for n = 1:length(files)
        img_name = files(n).name;
        ImagePathBoxCell{n} = fullfile(train_dir, img_name);
        
        if strcmp(img_name(1), '-') % junk images
            IDs(n) = -1;
            CAMs(n) = str2num(img_name(5));
        else
            IDs(n) = str2num(img_name(1:4));
            CAMs(n) = str2num(img_name(7));
        end
    end
    
    class_ids = IDs;
    unique_ids = unique(IDs);
    for id_index  = 1 : length(unique_ids)
        positions = IDs == unique_ids(id_index);
        % change the labelling
        class_ids(positions) = id_index;
    end
    
    save(save_path,...
        'class_ids',...
        'ImagePathBoxCell',...
        '-v7.3');