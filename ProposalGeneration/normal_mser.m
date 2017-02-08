function comp_infos = normal_mser(src, resizeRatio, minmaxRF, src_name)
    global param
    %this script use the normal version of mser extract regions
    %default: one channel
    
    imgPath = [param.workPath, '/temp/', src_name, '.jpg'];
    outputPath = [param.workPath, '/temp/', src_name, '.txt'];
        
    if(param.reuse_mser == false)
        img = rgb2gray(src);
        imwrite(img, imgPath, 'jpg');
        system(['/usr/local/opt/python/bin/python2.7 ./getMser.py ', imgPath, ' ', outputPath]);
    end
    [bbox8d, bbox] = localReadBoxes8d(outputPath);
    %bbox = localReadBoxes(outputPath);

    %mser
%     regions = detectMSERFeatures(img, 'ThresholdDelta', 0.1, ...
%         'RegionAreaRange', [5, size(img,1) * size(img,2) * 0.5], ...
%         'MaxAreaVariation', 1);
        
    % filter bbox
    rule1 = (bbox(:,3) ./ bbox(:,4)) < 2.5;
    rule2 = (bbox(:,4) ./ bbox(:,3)) < 5;
    rule3 = (max(bbox(:,3), bbox(:,4)) * resizeRatio) > minmaxRF;
    rule4 = bbox(:,3) .* bbox(:,4) > 20;
    bbox = bbox(rule1 & rule2 & rule3 & rule4,:);
    bbox8d = bbox8d(rule1 & rule2 & rule3 & rule4,:);
    bbox_ind = box_nms(bbox, bbox(:,3) .* bbox(:, 4), 0.8);
    bbox = bbox(bbox_ind, :);
    bbox8d = bbox8d(bbox_ind, :);
    
    %old code
    if false
        %record
        comp_infos = cell(size(bbox,1),1);
        for kk=1:size(bbox,1)
            box=bbox(kk,:);
            comp_infos{kk}.box=box;
            comp_infos{kk}.center=floor([box(1)+box(3)/2,box(2)+box(4)/2]);
        end
    end
    
    bbox = mat2cell(bbox, ones(size(bbox, 1), 1), 4);
    bbox8d = mat2cell(bbox8d, ones(size(bbox8d, 1), 1), 8);
    comp_infos = arrayfun(@localGetCompInfo, bbox, bbox8d);
end

function out = localGetCompInfo(x, y)
    out = cell(1);
    x = x{1};
    y = y{1};
    out{1}.box = x;
    out{1}.center=floor([x(1)+x(3)/2,x(2)+x(4)/2]);
    out{1}.box8d = y;
    
    x_arr = [y(1 : 2 : 8), y(1)];
    y_arr = [y(2 : 2 : 8), y(2)];
    
    edgeLens = ((x_arr(2 : 5) - x_arr(1 : 4)).^2 + (y_arr(2 : 5) - y_arr(1 : 4)).^2).^0.5;
    
    out{1}.shortEdge = min(edgeLens(1) + edgeLens(3), edgeLens(2) + edgeLens(4))/2;
    out{1}.longEdge = max(edgeLens(1) + edgeLens(3), edgeLens(2) + edgeLens(4))/2;
end

function [boxes8d, boxes4d] = localReadBoxes8d(resPath)
    fid = fopen(resPath, 'r');
    boxes8d = fscanf(fid, '%d %d %d %d %d %d %d %d', [8, inf]);
    fclose(fid);
    boxes8d = boxes8d';
    boxes4d = zeros(size(boxes8d, 1), 4);
    if(~isempty(boxes4d))
        boxes4d(:, 1) = min(boxes8d(:, 1 : 2 : 8), [], 2);
        boxes4d(:, 2) = min(boxes8d(:, 2 : 2 : 8), [], 2);
        boxes4d(:, 3) = max(boxes8d(:, 1 : 2 : 8), [], 2) - boxes4d(:, 1);
        boxes4d(:, 4) = max(boxes8d(:, 2 : 2 : 8), [], 2) - boxes4d(:, 2);
    else
        boxes8d = zeros(0, 8);
    end
end


function boxes = localReadBoxes(resPath)
    fid = fopen(resPath, 'r');
    boxes = fscanf(fid, '%d %d %d %d', [4, inf]);
    fclose(fid);
    boxes = boxes';
end