function r1 = compute_r1_multiCam(good_image, junk_image, index, queryCam, testCam)

r1 = ones(1, 6)-3;

% on the same camera
good_cam_now = queryCam;
ngood = length(junk_image)-1; % the number of other tracklets with the same id in this camera 
junk_image_now = [good_image; index(1)]; % this tracklets and tracklets in other cameras
good_image_now = setdiff(junk_image, index(1)); % the index of other tracklets
good_now = 0; 

%for n = 1:length(index) 
for n = 2:length(index) 
    if ngood == 0
        r1(good_cam_now) = -1;
        break;
    end
    
    flag = 0;
    if ~isempty(find(good_image_now == index(n), 1)) 
        flag = 1; % good image 
        good_now = good_now+1; 
    end
    if ~isempty(find(junk_image_now == index(n), 1))
        continue; % junk image 
    end
    if flag == 0
        r1(good_cam_now) = 0;
        break;
    end
    if flag == 1%good
        r1(good_cam_now) = 1;
        break;
    end 
end 

% search on every different camera
good_cam = testCam(good_image);
good_cam_uni = unique(good_cam); 

for k = 1:length(good_cam_uni)
    good_cam_now = good_cam_uni(k);

    ngood = length(find(good_cam == good_cam_now));
    
    pos_junk = find(good_cam ~= good_cam_now);
    junk_image_now = [junk_image; good_image(pos_junk)];

    pos_good = find(good_cam == good_cam_now);
    good_image_now = good_image(pos_good);
    
    for n = 1:length(index) 
        flag = 0;
        if ngood == 0
            r1(good_cam_now) = -1;
            break;
        end
        if ~isempty(find(good_image_now == index(n), 1)) 
            flag = 1; % good image 
        end
        if ~isempty(find(junk_image_now == index(n), 1))
            continue; % junk image 
        end

        if flag == 0
            r1(good_cam_now) = 0;
            break;
        end
        if flag == 1%good
            r1(good_cam_now) = 1;
            break;
        end 
    end 
end

end


