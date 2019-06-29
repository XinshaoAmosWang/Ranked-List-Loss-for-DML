function evaluate_clustering(name, embedding_dimension, distance)

if nargin < 1
    name = 'liftedstructsim_softmax_pair_m128_multilabel';
end

if nargin < 2
    embedding_dimension = 64;
end

if nargin < 3
    distance = 'euclidean';
end

K = 11316;
filename = sprintf('idx_kmeans_googlenet_%s_embed%d_%s.mat', name, embedding_dimension, distance);
if exist(filename, 'file')
    object = load(filename);
    idx = object.idx;
else
    % compute similarity
    load(sprintf('validation_googlenet_feat_matrix_%s_embed%d_baselr_1E4_gaussian2k.mat', ...
                                        name, embedding_dimension), 'fc_embedding');
    X = double(fc_embedding');

    % kmeans clustering
    fprintf('2d kmeans %d\n', K);
    opts = struct('maxiters', 1000, 'mindelta', eps, 'verbose', 1);
    [center, sse] = vgg_kmeans(X, K, opts);
    [idx_kmeans, d] = vgg_nearest_neighbour(X, center);

    % construct idx
    num = size(X, 2);
    idx = zeros(num, 1);
    for i = 1:K
        index = find(idx_kmeans == i);
        [~, ind] = min(d(index));
        cid = index(ind);
        idx(index) = cid;
    end

    fprintf('Number of clusters: %d\n', length(unique(idx)));
    save(sprintf('idx_kmeans_googlenet_%s_embed%d_%s.mat', name, embedding_dimension, distance), 'idx');
end

% evaluation
num_validation_classes = K;

% load ground truth from filenames
[image_ids, class_ids, superclass_ids, path_list] = ...
    textread('/cvgl/group/Ebay_Dataset/Ebay_test.txt', '%d %d %d %s',...
    'headerlines', 1);

num = numel(class_ids);
item_ids = cell(num,1);
for i = 1:num
    class_id = class_ids(i);                             
    item_ids{i} = num2str(class_id);
end 
assert(length(unique(item_ids)) == num_validation_classes);

% Given cluster assignment and the class names
%   Compute the three clustering metrics.
[NMI, RI, F1] = compute_clutering_metric(idx, item_ids);

fprintf('[method: %s, distance: %s] NMI: %.3f, RI: %.3f, F1: %.3f\n\n', ...
    name, distance, NMI, RI, F1);