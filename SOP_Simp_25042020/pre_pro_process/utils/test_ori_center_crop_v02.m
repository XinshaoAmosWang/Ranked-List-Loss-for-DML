% Workpace parameters:
% net:
% param.test_batch_size: 

% data_cell
% class_ids


test_features = forward_ori_center_crop_v02(net, TestImagePathBoxCell, ...
					param.test_batch_size, param.crop_padding, param.force_square_size,  blob_name);
test_num = size(test_features, 1);

%% cosine score matrix
% the higher value, the more similiar
test_features = bsxfun(@rdivide, test_features, sum(abs(test_features).^2,2).^(1/2));
score_matrix = test_features * test_features'; %(n, n)
% the smaller, the more similiar
score_matrix =  - score_matrix;
% set diagonal to very high number
score_matrix( 1 : test_num+1 : test_num*test_num) = inf;

%%

K = 3;
arr = zeros(K, 1);
arr(1) = compute_recall_at_K(score_matrix, 1, class_ids, test_num);
arr(2) = compute_recall_at_K(score_matrix, 10, class_ids, test_num);
arr(3) = compute_recall_at_K(score_matrix, 100, class_ids, test_num);


fin = fopen(record_file, 'a');
fprintf(fin, 'iter: %d, rank1: %.3f, rank10: %.3f, rank100: %.3f\n', ...
                                iter, arr(1), arr(2), arr(3));
fprintf('iter: %d, rank1: %.3f, rank10: %.3f, rank100: %.3f\n', ...
                                iter, arr(1), arr(2), arr(3));


%xinshao_test_clustering(test_features, class_ids, fin);

fclose(fin);




% compute recall@K
function recall = compute_recall_at_K(distance_matrix, K, class_ids, num)
	num_correct = 0;
	for i = 1 : num
	    this_gt_class_idx = class_ids(i);
	    this_row = distance_matrix(i,:);
	    [~, inds] = sort(this_row, 'ascend');
	    knn_inds = inds(1:K);
	    
	    knn_class_inds = class_ids(knn_inds);
	    
	    if sum(ismember(knn_class_inds, this_gt_class_idx)) > 0
	        num_correct = num_correct + 1;
	    end
	end
	recall = num_correct / num;
end
