

%% add necessary paths
%query_dir = 'dataset/query/';% query directory
%test_dir = 'dataset/bounding_box_test/';% database directory
%gt_dir = 'dataset/gt_bbox/'; % directory of hand-drawn bounding boxes
%addpath 'CM_Curve/' % draw confusion matrix

%% calculate query features
Hist_query = process_imagescell_net_blobname(net, queryImagesCell, param.test_batch_size, ...
                                            param.crop_padding, param.force_square_size,  param.cropped_size, blob_name);
Hist_query = Hist_query'; % transpose=> featureDim x nQuery
nQuery = size(Hist_query, 2);

%% calculate database features
Hist_test = process_imagescell_net_blobname(net, testImagesCell, param.test_batch_size, ...
                                            param.crop_padding, param.force_square_size,  param.cropped_size, blob_name);
Hist_test = Hist_test'; % transpose=> featureDim x nQuery
nTest = size(Hist_test, 2);

%% calculate features for multiple queries
Hist_max = zeros(size(Hist_test, 1),  length(multiQueryImagesCell));
Hist_avg = zeros(size(Hist_test, 1),  length(multiQueryImagesCell));
for n = 1 : length(multiQueryImagesCell)
    features_multiQueries = process_imagescell_net_blobname(net, multiQueryImagesCell{n}, param.test_batch_size, ...
                                            param.crop_padding, param.force_square_size,  param.cropped_size, blob_name);
     features_multiQueries = features_multiQueries';
    % feature L2 normalization
    sum_val = sqrt(sum(features_multiQueries.^2));
    sum_val = repmat(sum_val, [size(features_multiQueries,1), 1]);
    features_multiQueries = features_multiQueries./sum_val;
    % max or avg pooling of multiple queries
    Hist_max(:, n) = max(features_multiQueries, [], 2);
    Hist_avg(:, n) = mean(features_multiQueries, 2);
end
% another normalization
sum_val = sqrt(sum(Hist_max.^2));
sum_val = repmat(sum_val, [size(Hist_max,1), 1]);
sum_val2 = sqrt(sum(Hist_avg.^2));
sum_val2 = repmat(sum_val2, [size(Hist_avg,1), 1]);
Hist_max = Hist_max./sum_val;
Hist_avg = Hist_avg./sum_val2;


%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
ap_max = zeros(nQuery, 1); % average precision with MultiQ_max 
ap_avg = zeros(nQuery, 1); % average precision with MultiQ_avg 
ap_max_rerank  = zeros(nQuery, 1); % average precision with MultiQ_max + re-ranking 
ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, nTest);
CMC_max = zeros(nQuery, nTest);
CMC_avg = zeros(nQuery, nTest);
CMC_max_rerank = zeros(nQuery, nTest);

r1 = 0; % rank 1 precision with single query
r1_max = 0; % rank 1 precision with MultiQ_max
r1_avg = 0; % rank 1 precision with MultiQ_avg
r1_max_rerank = 0; % rank 1 precision with MultiQ_max + re-ranking
r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)

dist = sqdist(Hist_test, Hist_query); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
dist_max = sqdist(Hist_test, Hist_max); % distance calculation with MultiQ_max
dist_avg = sqdist(Hist_test, Hist_avg); % distance calculation with MultiQ_avg
dist_cos_max = (2-dist_max)./2; % cosine distance with MultiQ_max, used for re-ranking
knn = 1; % number of expanded queries. knn = 1 yields best result

for k = 1:nQuery
    %k
    % load groud truth for each query (good and junk)
    % images with the same ID but different camera from the query
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';
    % images neither good nor bad in terms of bbox quality
    junk_index1 = find(testID == -1);
    % images with the same ID and the same camera as the query
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); 
    junk_index = [junk_index1; junk_index2]';
    %tic
    score = dist(:, k);
    score_avg = dist_avg(:, k); 
    score_max = dist_max(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    [~, index_max] = sort(score_max, 'ascend'); % multiple queries by max pooling
    [~, index_avg] = sort(score_avg, 'ascend'); % multiple queries by avg pooling
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
    [ap_max(k), CMC_max(k, :)] = compute_AP(good_index, junk_index, index_max);% compute AP for MultiQ_max
    [ap_avg(k), CMC_avg(k, :)] = compute_AP(good_index, junk_index, index_avg);% compute AP for MultiQ_avg
    ap_pairwise(k, :) = compute_AP_multiCam(good_index, junk_index, index, queryCam(k), testCam); % compute pairwise AP for single query
    
    %%%%%%% re-ranking after "multiple queries with max pooling" %%%%%%%%%
    count = 0;
    score_cos_max = dist_cos_max(:, k);
    for i = 1:length(index_max) % index_max is the same with sorting images according to score_cos_max
        if ~isempty(find(junk_index == index_max(i), 1)) % a junk image
            continue;
        else
            count = count + 1;
            query_index = index_max(i);
            query_hist_new = single(Hist_test(:, query_index));% expanded query
            query_hist_new = repmat(query_hist_new, [1, size(Hist_test, 2)]);
            score_new = sum(query_hist_new.*Hist_test);
            score_cos_max = score_cos_max + score_new'./(count+1); % update score
            if count == knn % will break if "knn" queries are expanded
                break;
            end
        end
    end
    [~, index_max_rerank] = sort(score_cos_max, 'descend');
    [ap_max_rerank(k), CMC_max_rerank(k, :)] = compute_AP(good_index, junk_index, index_max_rerank); % compute AP for MultiQ_max + re-rank
    %%%%%%%%%%%%%%%%%%%%%%%% re-ranking %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%% calculate pairwise r1 precision %%%%%%%%%%%%%%%%%%%%
    r1_pairwise(k, :) = compute_r1_multiCam(good_index, junk_index, index, queryCam(k), testCam); % pairwise rank 1 precision with single query
    %%%%%%%%%%%%%% calculate r1 precision %%%%%%%%%%%%%%%%%%%%
end
CMC_max_rerank = mean(CMC_max_rerank);
CMC_max = mean(CMC_max);
CMC_avg = mean(CMC_avg);
CMC = mean(CMC);
%% print result
fprintf('single query:                                   mAP = %f, r1 precision = %f\r ', mean(ap), CMC(1));
fprintf('multiple queries with avg pooling:              mAP = %f, r1 precision = %f\r ', mean(ap_avg), CMC_avg(1));
fprintf('multiple queries with max pooling:              mAP = %f, r1 precision = %f\r ', mean(ap_max), CMC_max(1));
fprintf('multiple queries with max pooling + re-ranking: mAP = %f, r1 precision = %f\r\n', mean(ap_max_rerank), CMC_max_rerank(1));

fin = fopen(record_file, 'a');
fprintf(fin, 'iter: %d, rank1: %f, rank5: %f, rank10: %f, rank20: %f, mAP: %f\n',...
    iter, CMC_arr(1), CMC_arr(5), CMC_arr(10), CMC_arr(20), mAP);
fprintf(fin, 'single query:                                   mAP = %f, r1 precision = %f\r ', mean(ap), CMC(1));
fprintf(fin, 'multiple queries with avg pooling:              mAP = %f, r1 precision = %f\r ', mean(ap_avg), CMC_avg(1));
fprintf(fin, 'multiple queries with max pooling:              mAP = %f, r1 precision = %f\r ', mean(ap_max), CMC_max(1));
fprintf(fin, 'multiple queries with max pooling + re-ranking: mAP = %f, r1 precision = %f\r\n', ...
    mean(ap_max_rerank), CMC_max_rerank(1));
fclose(fin);








%[ap_CM, r1_CM] = draw_confusion_matrix(ap_pairwise, r1_pairwise, queryCam);
%fprintf('average of confusion matrix with single query:  mAP = %f, r1 precision = %f\r\n', (sum(ap_CM(:))-sum(diag(ap_CM)))/30, (sum(r1_CM(:))-sum(diag(r1_CM)))/30);

%% plot CMC curves
%figure;
%s = 50;
%CMC_curve = [CMC_max_rerank; CMC_max; CMC_avg; CMC ];
%plot(1:s, CMC_curve(:, 1:s));




