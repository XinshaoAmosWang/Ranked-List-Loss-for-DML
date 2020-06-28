% Workpace parameters:
% net:
% param.test_batch_size: 

% data_cell
% class_ids

% Query: images_num: 14,218, class_num: 3,985 
% Gallery: images_num: 12,612, class_num: 3,985 

test_features = forward_ori_center_crop_v03(net, TestImagePathBoxCell, ...
					param.test_batch_size, ...
					param.crop_padding, param.force_square_size, param.cropped_size, ...
					blob_name);

query_num = 14218;
gallery_num = 12612;

%% cosine score matrix
% the higher value, the more similiar
test_features = bsxfun(@rdivide, test_features, sum(abs(test_features).^2,2).^(1/2));
query_features = test_features( 1:query_num, :);
gallery_features = test_features( query_num + 1 : end, :);
score_matrix = query_features * gallery_features'; %(query_num, gallery_num)
% the smaller, the more similiar
score_matrix =  - score_matrix;

%%
K = 6;
arr = zeros(K, 1);
arr(1) = compute_recall_at_K(score_matrix, 1, class_ids, query_num, gallery_num);
arr(2) = compute_recall_at_K(score_matrix, 10, class_ids, query_num, gallery_num);
arr(3) = compute_recall_at_K(score_matrix, 20, class_ids, query_num, gallery_num);
arr(4) = compute_recall_at_K(score_matrix, 30, class_ids, query_num, gallery_num);
arr(5) = compute_recall_at_K(score_matrix, 40, class_ids, query_num, gallery_num);
arr(6) = compute_recall_at_K(score_matrix, 50, class_ids, query_num, gallery_num);


fin = fopen(record_file, 'a');
fprintf(fin, 'iter: %d, rank1: %.3f, rank10: %.3f, rank20: %.3f, rank30: %.3f, rank40: %.3f, rank50: %.3f\n', ...
                                iter, arr(1), arr(2), arr(3), arr(4), arr(5), arr(6));
fprintf('iter: %d, rank1: %.3f, rank10: %.3f, rank20: %.3f, rank30: %.3f, rank40: %.3f, rank50: %.3f\n', ...
			iter, arr(1), arr(2), arr(3), arr(4), arr(5), arr(6));


%xinshao_test_clustering(test_features, class_ids, fin);

fclose(fin);




% compute recall@K
% score_matrix: size(query_num, gallery_num)
% class_ids: query_num + gallery_num in order
function recall = compute_recall_at_K(distance_matrix, K,...
						 class_ids, query_num, gallery_num)
	num_correct = 0;
	for i = 1 : query_num
	    this_gt_class_idx = class_ids(i);
	    this_row = distance_matrix(i,:); % 1 x gallery_num
	    [~, inds] = sort(this_row, 'ascend');
	    knn_inds = inds(1:K); % first K indexes has K smallest distance
		
	    % the class_ids of the first K gallery images
	    knn_class_inds = class_ids(query_num + knn_inds);
	    
	    if sum(ismember(knn_class_inds, this_gt_class_idx)) > 0
	        num_correct = num_correct + 1;
	    end
	end
	% compute recall
	recall = num_correct / query_num;
end
