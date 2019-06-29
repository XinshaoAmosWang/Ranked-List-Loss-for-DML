function evaluate_recall(embedding_dimension)

fprintf('embedding_dimension %d\n', embedding_dimension);

name = 'liftedstructsim_softmax_pair_m128_multilabel';
feature_filename = ...
    sprintf('validation_googlenet_feat_matrix_%s_embed%d_baselr_1E4_gaussian2k.mat', ...
    name, embedding_dimension);

try
    load(feature_filename, 'fc_embedding');
catch
    error('filename is non existent.\n');
end

features = fc_embedding;

dims = size(features);
assert(dims(2) == embedding_dimension);
assert(dims(1) == 60502);

%D = squareform(pdist(features, 'euclidean'));
D2 = distance_matrix(features);

[image_ids, class_ids, superclass_ids, path_list] = ...
    textread('/cvgl/group/Ebay_Dataset/Ebay_test.txt', '%d %d %d %s',...
    'headerlines', 1); 

% set diagonal to very high number
assert(dims(1) == length(class_ids));
num = dims(1);
D = sqrt(abs(D2));
D(1:num+1:num*num) = inf;

for K = [1, 10, 100, 1000]
    compute_recall_at_K(D, K, class_ids, num);
end
    
disp('done');

% compute pairwise distance matrix
function D = distance_matrix(X)

m = size(X, 1);
t = ones(m, 1);
x = zeros(m, 1);
for i = 1:m
    n = norm(X(i,:));
    x(i) = n * n;
end

D = x * t' + t * x' - 2 * X * X';

% compute recall@K
function recall = compute_recall_at_K(D, K, class_ids, num)

num_correct = 0;
for i = 1 : num
    this_gt_class_idx = class_ids(i);
    this_row = D(i,:);
    [~, inds] = sort(this_row, 'ascend');
    knn_inds = inds(1:K);
    
    knn_class_inds = class_ids(knn_inds);
    
    if sum(ismember(knn_class_inds, this_gt_class_idx)) > 0
        num_correct = num_correct + 1;
    end
end
recall = num_correct / num;
fprintf('K: %d, Recall: %.3f\n', K, recall);