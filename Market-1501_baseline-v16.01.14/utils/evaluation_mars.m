function [CMC, map, r1_pairwise, ap_pairwise] = evaluation_mars(dist, label_gallery, label_query, cam_gallery, cam_query)

%junk0 = find(label_gallery == -1);
junk0 = find(label_gallery == 0);

ap = zeros(size(dist, 2), 1);
CMC = [];
r1_pairwise = zeros(size(dist, 2), 6);% pairwise rank 1 precision  
ap_pairwise = zeros(size(dist, 2), 6); % pairwise average precision

for k = 1:size(dist, 2)
    % the score of each query, lable ,cam
    score = dist(:, k);
    q_label = label_query(k);
    q_cam = cam_query(k);

    % same label 
    pos = find(label_gallery == q_label);
    % same label, different cameras 
    pos2 = cam_gallery(pos) ~= q_cam;
    % same label, same camera
    pos3 = cam_gallery(pos) == q_cam;
    
    % Selected index: same label, different cameras
    good_image = pos(pos2);
    
    junk = pos(pos3); % same label in the same camera
    junk_image = [junk0; junk];% label == -1 & same label in the same camera

    [~, index] = sort(score, 'ascend');
    [ap(k), CMC(:, k)] = compute_AP(good_image, junk_image, index);
    ap_pairwise(k, :) = compute_AP_multiCam(good_image, junk, index, q_cam, cam_gallery); % compute pairwise AP for single query
    r1_pairwise(k, :) = compute_r1_multiCam(good_image, junk, index, q_cam, cam_gallery); % pairwise rank 1 precision with single query
end
CMC = sum(CMC, 2)./size(dist, 2);
CMC = CMC';
map = sum(ap)/length(ap);