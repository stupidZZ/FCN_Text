function orientation = estimateOrientation(img, region, region_comp_infos, orient_param)
    global param
    
    %% return if region_comp_infos is empty
    if isempty(region_comp_infos)
        orientation = NaN;
        return
    end
    
    if length(region_comp_infos) == 1
        orientation = 0;
        return;
    end
    
    if(orient_param.minOrientation == orient_param.maxOrientation)
        orientation = 0;
        return;
    end
    
    box8d = cell(length(region_comp_infos), 1);
    for i = 1 : length(region_comp_infos)
        box8d{i} = ConvertBox4dToBox8d(region_comp_infos{i}.box);
    end
    
    if(false && param.debug)
        debug_box = zeros(length(region_comp_infos), 4);
        for d_i = 1 : length(region_comp_infos)
            debug_box(d_i,:) = region_comp_infos{d_i}.box;
        end
        show_bbox(img, debug_box);
    end
        
    region_cx = mean(region(:,2));
    region_cy = mean(region(:,1));

    orientations = [orient_param.minOrientation : orient_param.orientationInterval : orient_param.maxOrientation];
    hitBoxCount = zeros(length(orientations), 1);
    
    
    y1_arr = zeros(length(region_comp_infos), 1);
    y2_arr = zeros(length(region_comp_infos), 1);
    for i = 1 : length(orientations)
        
        for j = 1 : length(region_comp_infos)
            rotate_box8d = getRotateBox8D(box8d{j}, orientations(i), region_cx, region_cy);
            
            y1_arr(j) = min(rotate_box8d(:,1));
            y2_arr(j) = max(rotate_box8d(:,1));
            
            y_offest_arr = y2_arr(j) - y1_arr(j) + 1;
            y1_arr(j) = y1_arr(j) + floor(0.3 * y_offest_arr);
            y2_arr(j) = y2_arr(j) - floor(0.3 * y_offest_arr);
        end
        
        min_y_arr = min(y1_arr);
        y_map = zeros(max(y2_arr) - min_y_arr + 1, 1);
        for j = 1 : length(region_comp_infos)
            y_map(y1_arr(j) - min_y_arr + 1 : y2_arr(j) - min_y_arr + 1) = ...
                y_map(y1_arr(j) - min_y_arr + 1 : y2_arr(j) - min_y_arr + 1) + 1;
        end
        
        hitBoxCount(i) = max(y_map);
        
        if false && param.debug
            imshow(img);
            hold on;
            for k = 1 : length(region_comp_infos)
                cy = mean(box8d{k}(:,1));
                cx = mean(box8d{k}(:,2));

                plot(cx,cy, '*', 'color', 'y');
                
                rotate_box8d = getRotateBox8D(box8d{k}, orientations(i), region_cx, region_cy);
                cy = mean(rotate_box8d(:,1));
                cx = mean(rotate_box8d(:,2));
                plot(cx,cy, '*', 'color', 'r');
                
                plot([rotate_box8d(:, 2); rotate_box8d(1, 2)], [rotate_box8d(:, 1); rotate_box8d(1, 1)]);
            end
            hold off;
        end
    end
    
    max_hit_box = max(hitBoxCount);
    
    %% calc_continues
    continueScore = zeros(length(hitBoxCount), 1);
    if(hitBoxCount(length(hitBoxCount)) == max_hit_box)
        continueScore(length(hitBoxCount)) = 1;
    end
    
    for i = length(hitBoxCount) - 1 : -1 : 1
        if(hitBoxCount(i) == max_hit_box)
            continueScore(i) = continueScore(i+1) + 1;
        else
            continueScore(i) = 0;
        end
    end
    
    left_most = find(continueScore == max(continueScore));
    if(length(left_most) > 1)
        [~, middlest] = min(abs(left_most - length(hitBoxCount)));
        left_most = left_most(middlest);
    end
    
    right_most = left_most;
    len_max_hit_box = 1;
    
    while true
        i = left_most - 1;
        if(i == 0)
            i = length(hitBoxCount);
        end
        if(hitBoxCount(i) == max_hit_box && right_most ~= i)
            left_most = i;
            len_max_hit_box = len_max_hit_box + 1;
        else
            break;
        end
    end
    
    while true
        i = right_most + 1;
        if(i >= length(hitBoxCount))
            i = 1;
        end
        if(hitBoxCount(i) == max_hit_box && left_most ~= i)
            right_most = i;
            len_max_hit_box = len_max_hit_box + 1;
        else
            break;
        end
    end
    
    max_orientation_idx = left_most + round(len_max_hit_box/2);
    if max_orientation_idx > length(hitBoxCount)
        max_orientation_idx = max_orientation_idx - length(hitBoxCount);
    end
    orientation = orientations(max_orientation_idx);
    
    if param.debug
        imshow(img);
        hold on;
        for k = 1 : length(region_comp_infos)
            cy = mean(box8d{k}(:,1));
            cx = mean(box8d{k}(:,2));

            plot(cx,cy, '*', 'color', 'y');
            
            rotate_box8d = getRotateBox8D(box8d{k}, orientation, region_cx, region_cy);

            cy = mean(rotate_box8d(:,1));
            cx = mean(rotate_box8d(:,2));

            plot(cx,cy, '*', 'color', 'r');
        end
        hold off;
    end
end

