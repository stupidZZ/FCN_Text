function nms_idx = box_nms(boxes, scores, area_threashold)
    [~, sort_idx] = sort(scores, 'descend');
    boxes = boxes(sort_idx, :);
    area = boxes(:,3) .* boxes(:,4);
    
    nms_flag = ones(size(boxes, 1), 1);
    for i = 1 : size(boxes,1)
        if(nms_flag(i) == 0)
            continue;
        end
        
        int_area = rectint(boxes(i, :), boxes)';
        area_ratio = int_area ./ (area + area(i) - int_area);
        area_ratio(1 : i) = 0;
        nms_flag(area_ratio > area_threashold) = 0;
    end
    
    nms_idx = sort_idx(logical(nms_flag));
end

