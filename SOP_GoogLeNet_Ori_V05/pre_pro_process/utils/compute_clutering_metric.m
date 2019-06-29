function [NMI, RI, F] = compute_clutering_metric(idx, item_ids)

N = length(idx);

% cluster centers
centers = unique(idx);
num_cluster = numel(centers);
fprintf('Number of clusters: %d\n', num_cluster);

% count the number of objects in each cluster
count_cluster = zeros(1, num_cluster);
for i = 1:num_cluster
    count_cluster(i) = numel(find(idx == centers(i)));
end

% build a mapping from item_id to item index
keys = unique(item_ids);
num_item = numel(keys);
values = 1:num_item;
item_map = containers.Map(keys, values);

% count the number of objects of each item
count_item = zeros(1, num_item);
for i = 1:N
    index = item_map(item_ids{i});
    count_item(index) = count_item(index) + 1;
end

% compute purity
purity = 0;
for i = 1:num_cluster
    member = find(idx == centers(i));
    member_ids = item_ids(member);
    
    count = zeros(1, num_item);
    for j = 1:numel(member)
        index = item_map(member_ids{j});
        count(index) = count(index) + 1;
    end
    purity = purity + max(count);
end
purity = purity / N;
fprintf('Purity is %f\n', purity);

% compute Normalized Mutual Information (NMI)
count_cross = zeros(num_cluster, num_item);
for i = 1:N
    index_cluster = find(idx(i) == centers);
    index_item = item_map(item_ids{i});
    count_cross(index_cluster, index_item) = ...
        count_cross(index_cluster, index_item) + 1;
end

% mutual information
I = 0;
for k = 1:num_cluster
    for j = 1:num_item
        if count_cross(k, j) > 0
            s = count_cross(k, j) / N * log(N * count_cross(k, j) / (count_cluster(k) * count_item(j)));
            I = I + s;
        end
    end
end
fprintf('Mutual information is %f\n', I);

% entropy
H_cluster = 0;
for k = 1:num_cluster
    s = -count_cluster(k) / N * log(count_cluster(k) / N);
    H_cluster = H_cluster + s;
end
fprintf('Entropy cluster is %f\n', H_cluster);

H_item = 0;
for j = 1:num_item
    s = -count_item(j) / N * log(count_item(j) / N);
    H_item = H_item + s;
end
fprintf('Entropy item is %f\n', H_item);

NMI = 2 * I / (H_cluster + H_item);
fprintf('NMI is %f\n', NMI);

% compute True Positive (TP) plus False Positive (FP)
tp_fp = 0;
for k = 1:num_cluster
    if count_cluster(k) > 1
        tp_fp = tp_fp + nchoosek(count_cluster(k), 2);
    end
end

% compute True Positive (TP)
tp = 0;
for k = 1:num_cluster
    member = find(idx == centers(k));
    member_ids = item_ids(member);
    
    count = zeros(1, num_item);
    for j = 1:numel(member)
        index = item_map(member_ids{j});
        count(index) = count(index) + 1;
    end
    
    for i = 1:num_item
        if count(i) > 1
            tp = tp + nchoosek(count(i), 2);
        end
    end
end
fprintf('TP is %d\n', tp);

% False Positive (FP)
fp = tp_fp - tp;
fprintf('FP is %d\n', fp);

% compute False Negative (FN)
count = 0;
for j = 1:num_item
    if count_item(j) > 1
        count = count + nchoosek(count_item(j), 2);
    end
end
fn = count - tp;
fprintf('FN is %d\n', fn);

% compute True Negative (TN)
tn = N*(N-1)/2 - tp - fp - fn;
fprintf('TN is %d\n', tn);

% compute RI
RI = (tp + tn) / (tp + fp + fn + tn);
fprintf('RI is %f\n', RI);

% compute F measure
P = tp / (tp + fp);
R = tp / (tp + fn);
fprintf('Precision is %f\n', P);
fprintf('Recall is %f\n', R);
beta = 1;
F = (beta*beta + 1) * P * R / (beta*beta * P + R);
fprintf('F_beta is %f with beta %.1f\n', F, beta);