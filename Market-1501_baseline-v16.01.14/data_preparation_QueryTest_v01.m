
%% add necessary paths
root = '/home/xinshao/Papers_Projects/Data/Market-1501-v15.09.15';
test_dir = fullfile(root, 'bounding_box_test/');% test_dir directory
query_dir = fullfile(root, 'query/');% query_dir directory
gt_dir = fullfile(root, 'gt_bbox/'); % directory of hand-drawn bounding boxes

%% calculate the ID and camera for database images
test_files = dir([test_dir '*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
testImagesCell = cell(length(test_files), 1);
%
    for n = 1:length(test_files)
        img_name = test_files(n).name;
        testImagesCell{n} = fullfile(test_dir, img_name);
        
        if strcmp(img_name(1), '-') % junk images
            testID(n) = -1;
            testCAM(n) = str2num(img_name(5));
        else
            testID(n) = str2num(img_name(1:4));
            testCAM(n) = str2num(img_name(7));
        end
    end
    %save('data/testID.mat', 'testID');
    %save('data/testCAM.mat', 'testCAM');
    
%% calculate the ID and camera for query images
query_files = dir([query_dir '*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
queryImagesCell = cell(length(query_files), 1);

    for n = 1:length(query_files)
        img_name = query_files(n).name;
        queryImagesCell{n} = fullfile(query_dir, img_name);
        
        if strcmp(img_name(1), '-') % junk images
            queryID(n) = -1;
            queryCAM(n) = str2num(img_name(5));
        else
            queryID(n) = str2num(img_name(1:4));
            queryCAM(n) = str2num(img_name(7));
        end
    end
    %save('data/queryID.mat', 'queryID');
    %save('data/queryCAM.mat', 'queryCAM');
    %% for multiple queries of each person
    multiQueryImagesCell = cell(length(query_files), 1);
    for n = 1:length(query_files)
        %n
        img_name = query_files(n).name;
        gt_files = dir([gt_dir img_name(1:7) '*.jpg']);
        
        multiQueryImagesCell{n} = cell(length(gt_files), 1);
        for m = 1:length(gt_files)
            img_path = fullfile(gt_dir, gt_files(m).name);
            multiQueryImagesCell{n}{m} = img_path;
        end
    end
%%
    save('Market1501_evaluation.mat',...
        'testID',...
        'testCAM',...
        'testImagesCell',...
        ...    
        'queryID',...
        'queryCAM',...
        'queryImagesCell',...
        'multiQueryImagesCell', ...
        '-v7.3');