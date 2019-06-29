% Workpace parameters:
% net:
% param.test_batch_size: 

% data_cell
% class_ids

test_features = forward_ori_center_crop_v02(net, TestImagePathBoxCell, ...
					param.test_batch_size, param.crop_padding, param.force_square_size );
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
K = 6;
arr = zeros(K, 1);
for i = 1 : K
    arr(i) = compute_recall_at_K(score_matrix, power(2, i-1), class_ids, test_num);
end


fin = fopen(record_file, 'a');
fprintf(fin, 'iter: %d, rank1: %.3f, rank2: %.3f, rank4: %.3f, rank8: %.3f, rank16: %.3f, rank32: %.3f\n', ...
				iter, arr(1), arr(2), arr(3), arr(4), arr(5), arr(6));
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
