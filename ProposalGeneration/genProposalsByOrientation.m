function proposals = genProposalsByOrientation(img, region, regionPerim, estimated_orientation, region_comp_infos)
    global param;
    
    if(isnan(estimated_orientation))
        proposals = zeros(0, 9);
        return
    end
    
    [comp_cluster_ind, ~] = compCluster(img, region_comp_infos, ...
                                                    estimated_orientation, ...
                                                    param.minCompHeightSimilarity, ...
                                                    param.maxCompOrientationDiff, ...
                                                    param.minIoUDiff);   

    clusterIdx = unique(comp_cluster_ind);
    clusterCount = length(clusterIdx);
    proposals = zeros(clusterCount, 9);
    confident_idx = false(clusterCount, 1);
    
    for c = 1 : clusterCount
        clusterCompInfos = region_comp_infos(comp_cluster_ind == clusterIdx(c));
        if(length(clusterCompInfos) > 0)
            confident_idx(c) = true;
            proposals(c, 1 : 8) = getProposal(img, region, regionPerim, clusterCompInfos, estimated_orientation);
            proposals(c, 9) = estimated_orientation;
        end
    end
    
    proposals = proposals(confident_idx, :);
    
    %% show proposals
    if true && param.debug
        imshow(img);
        hold on;
        for i = 1 : clusterCount
            x_arr = [proposals(i, 1 : 2 : 8), proposals(i, 1)];
            y_arr = [proposals(i, 2 : 2 : 8), proposals(i, 2)];

            plot(x_arr, y_arr, 'color', rand(3,1));
        end
        hold off;
    end
    
    %% show clusters
    if false && param.debug
        debug_bbox_gathered = zeros(0, 4);
        color_gathered = zeros(0, 3);
        for c = 1: clusterCount
            clusterCompInfos = region_comp_infos(comp_cluster_ind == clusterIdx(c));
            debug_bbox = zeros(length(clusterCompInfos), 4);
            color = repmat(rand(1,3), length(clusterCompInfos), 1);

            for d_i = 1 : length(clusterCompInfos)
                debug_bbox(d_i,:) = clusterCompInfos{d_i}.box;
            end
            
            debug_bbox_gathered = cat(1, debug_bbox_gathered, debug_bbox);
            color_gathered = cat(1, color_gathered, color);
        end
        show_bbox(img, debug_bbox_gathered, color_gathered);
    end
end

function [comp_cluster_ind, clusterCount] = compCluster(img, region_comp_infos, estimated_orientation, minCompHeightSimilarity, maxCompOrientationDiff, minIoUDiff)
    global param;
    
    compCount = length(region_comp_infos);
    
    if(compCount == 0)
        comp_cluster_ind = zeros(0, 1);
        clusterCount = 0;
        return;
    end
    
    if(compCount == 1)
        comp_cluster_ind = ones(1,1);
        clusterCount = 1;
        return;
    end
    
    comp_cluster_ind = zeros(compCount, 1, 'uint16');
    clusterCount = 0;
    
    box4d = zeros(compCount, 4);
    heights = zeros(compCount, 1);
    
    boxCenter = zeros(compCount, 2);
    boxCenter_U = zeros(compCount, 2);
    boxCenter_D = zeros(compCount, 2);
    boxCenter_L = zeros(compCount, 2);
    boxCenter_R = zeros(compCount, 2);

    for i = 1 : compCount
        box4d(i, :) = region_comp_infos{i}.box;
        [rectX, rectY, ~, ~, sideLenght] = computeMergedBox(box4d(i, :));
        [~, heights(i)] = findWidthAndHeight(rectX, rectY, sideLenght, estimated_orientation);
        
        boxCenter(i, 2) = round(box4d(i, 1) + 0.5*box4d(i, 3));
        boxCenter(i, 1) = round(box4d(i, 2) + 0.5*box4d(i, 4));
        
        boxCenter_U(i, 2) = round(box4d(i, 1) + 0.5*box4d(i, 3));
        boxCenter_U(i, 1) = box4d(i, 2);
        
        boxCenter_D(i, 2) = round(box4d(i, 1) + 0.5*box4d(i, 3));
        boxCenter_D(i, 1) = box4d(i,2) + box4d(i,4) - 1;
        
        boxCenter_L(i, 2) = box4d(i,1);
        boxCenter_L(i, 1) = round(box4d(i, 2) + 0.5*box4d(i, 4));
        
        boxCenter_R(i, 2) = box4d(i,1) + box4d(i,3) - 1;
        boxCenter_R(i, 1) = round(box4d(i, 2) + 0.5*box4d(i, 4));
    end
    
    heightSimilarity = zeros(compCount);
    IoUDiff = zeros(compCount);
    orientationDiffOfBox = zeros(compCount);
    dists = zeros(compCount);

    for i = 1 : compCount
        
        %% Compute height similarity
        heightSimilarity(i,:) = min( ...
                                    min(box4d(i, 4), box4d(:, 4)) ./ max(box4d(i, 4), box4d(:, 4)), ...
                                    min(box4d(i, 3), box4d(:, 3)) ./ max(box4d(i, 3), box4d(:, 3)));
       %% Compute IoU diff
        intArea = rectint(box4d(i,:), box4d);
        unionArea = box4d(i,3) * box4d(i,4) + box4d(:, 3).* box4d(:, 4);
        IoUDiff(i, :) = intArea ./ unionArea';
        
        %% Compute orientation diff of box
        orientationDiff_U = computeOrientationDiff(boxCenter_U, boxCenter_U(i,:), estimated_orientation);
        orientationDiff_D = computeOrientationDiff(boxCenter_D, boxCenter_D(i,:), estimated_orientation);
        orientationDiff_L = computeOrientationDiff(boxCenter_L, boxCenter_L(i,:), estimated_orientation);
        orientationDiff_R = computeOrientationDiff(boxCenter_R, boxCenter_R(i,:), estimated_orientation);
        orientationDiffOfBox(i, :) = min(min(orientationDiff_U, orientationDiff_D), ...
                                    min(orientationDiff_L, orientationDiff_R)); 
                                
        %% Compute dist
        dists(i, :) = ((boxCenter(:, 1) - boxCenter(i, 1)).^2 + (boxCenter(:, 2) - boxCenter(i,2)).^2).^0.5;
    end
    
    if true && param.debug
        imshow(img);
        for i = 1 : compCount
            rectangle('position', box4d(i, :), 'edgecolor','y');
        end
    end
    
    for i = 1 : compCount
        if comp_cluster_ind(i) == 0 
            clusterCount = clusterCount + 1;
            current_cluster_comp_ind = false(compCount, 1);
            current_cluster_comp_ind(i) = 1;
            
            isUpdate = true;
            while isUpdate
                isUpdate = false;
                
                %% Compute boxCenter of merged bbox
                mergedBoxCenter_U = [mean(box4d(current_cluster_comp_ind, 2)), ...
                                        mean(box4d(current_cluster_comp_ind, 1) + 0.5*box4d(current_cluster_comp_ind, 3))];

                mergedBoxCenter_D = [mean(box4d(current_cluster_comp_ind, 2) + box4d(current_cluster_comp_ind, 4) - 1), ...
                                        mean(box4d(current_cluster_comp_ind, 1) + 0.5*box4d(current_cluster_comp_ind, 3))];
                                        
                mergedBoxCenter_L = [mean(box4d(current_cluster_comp_ind, 2) + 0.5*box4d(current_cluster_comp_ind, 4)), ...
                                        mean(box4d(current_cluster_comp_ind, 1))];
                                    
                mergedBoxCenter_R = [mean(box4d(current_cluster_comp_ind, 2) + 0.5*box4d(current_cluster_comp_ind, 4)), ...
                                        mean(box4d(current_cluster_comp_ind, 1) + box4d(current_cluster_comp_ind, 3) - 1)];
                
                %% Compute orientation diff
                orientationDiff_U = computeOrientationDiff(boxCenter_U, mergedBoxCenter_U, estimated_orientation);
                orientationDiff_D = computeOrientationDiff(boxCenter_D, mergedBoxCenter_D, estimated_orientation);
                orientationDiff_L = computeOrientationDiff(boxCenter_L, mergedBoxCenter_L, estimated_orientation);
                orientationDiff_R = computeOrientationDiff(boxCenter_R, mergedBoxCenter_R, estimated_orientation);

                orientationDiff = min(min(orientationDiff_U, orientationDiff_D), min(orientationDiff_L, orientationDiff_R))';
                
                heightSimilarity_rule = heightSimilarity(current_cluster_comp_ind,:) > minCompHeightSimilarity;
                orientationDiff_rule = orientationDiff < maxCompOrientationDiff;
                IoUDiff_rule = IoUDiff(current_cluster_comp_ind, :) > minIoUDiff;
                orientationDiffOfBox_rule = heightSimilarity_rule ...
                                            & (orientationDiffOfBox(current_cluster_comp_ind, :) < maxCompOrientationDiff) ...
                                            & (dists(current_cluster_comp_ind, :) < param.maxDistRatio * mean(box4d(current_cluster_comp_ind, 4)));
                
                spatial_rule = ((sum(heightSimilarity_rule, 1) > 0) & sum(orientationDiffOfBox_rule, 1) > 0) ...
                                | (sum(IoUDiff_rule, 1) > 0) ...
                                | (orientationDiff_rule & (sum(heightSimilarity_rule, 1) > 0));
                %spatial_rule = sum((heightSimilarity_rule & orientationDiff_rule) | IoUDiff_rule, 1) > 0;
                
                satisfiedComp = spatial_rule' ...
                            & (~current_cluster_comp_ind);
                
                satisfiedIdx = find(satisfiedComp);
                if(~isempty(satisfiedIdx))
                    isUpdate = true;
                    current_cluster_comp_ind(satisfiedIdx) = 1;
%                     x1_tmp = min(min(box4d(satisfiedIdx, 1)), ...
%                                  mergedBox(1));
%                     x2_tmp = max(max(box4d(satisfiedIdx, 1) + box4d(satisfiedIdx, 3) - 1), ...
%                                  mergedBox(1) + mergedBox(3) - 1);
%                     y1_tmp = min(min(box4d(satisfiedIdx, 2)), ...
%                                  mergedBox(2));
%                     y2_tmp = max(max(box4d(satisfiedIdx, 2) + box4d(satisfiedIdx, 4) - 1), ...
%                                  mergedBox(2) + mergedBox(4) - 1);
%                     mergedBox(1) = x1_tmp;
%                     mergedBox(2) = y1_tmp;
%                     mergedBox(3) = x2_tmp - x1_tmp + 1;
%                     mergedBox(4) = y2_tmp - y1_tmp + 1;


                    if true &&  param.debug
                        clusterCompInfos = region_comp_infos(current_cluster_comp_ind);
                        color = zeros(length(clusterCompInfos), 3);
                        color(:, 2) = 1;
                        color(1, :) = [1, 0, 0];
                        imshow(img);
                        for d_i = 1 : length(clusterCompInfos)
                           rectangle('position', clusterCompInfos{d_i}.box, 'edgecolor', color(d_i,:));
                        end
                    end
                end
            end
            comp_cluster_ind(current_cluster_comp_ind) = clusterCount;
        end
    end
end
