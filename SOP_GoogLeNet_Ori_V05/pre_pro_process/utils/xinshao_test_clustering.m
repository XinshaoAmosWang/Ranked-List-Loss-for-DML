function xinshao_test_clustering(test_features, class_ids, fin)



num_validation_classes = length(unique(class_ids));
fc_embedding = test_features;

    X = double(fc_embedding');

    % kmeans clustering
    fprintf('2d kmeans %d\n', num_validation_classes);
    opts = struct('maxiters', 1000, 'mindelta', eps, 'verbose', 1);
    [center, sse] = vgg_kmeans(X, num_validation_classes, opts);
    [idx_kmeans, d] = vgg_nearest_neighbour(X, center);

    % construct idx
    num = numel(class_ids);
    idx = zeros(num, 1);
    for i = 1:num_validation_classes
        index = find(idx_kmeans == i);
        [~, ind] = min(d(index));
        cid = index(ind);
        idx(index) = cid;
    end

    fprintf('Number of clusters: %d\n', length(unique(idx)));
    %save(sprintf('idx_kmeans_googlenet_%s_embed%d_%s.mat', name, embedding_dimension, distance), 'idx');




% evaluation
item_ids = cell(num,1);
for i = 1:num
    class_id = class_ids(i);                             
    item_ids{i} = num2str(class_id);
end 
assert(length(unique(item_ids)) == num_validation_classes);



% Given cluster assignment and the class names
%   Compute the three clustering metrics.
[NMI, RI, F1] = compute_clutering_metric(idx, item_ids);

fprintf('NMI: %.3f, RI: %.3f, F1: %.3f\n\n', ...
    NMI, RI, F1);

fprintf(fin, 'NMI: %.3f, RI: %.3f, F1: %.3f\n\n', ...
    NMI, RI, F1);