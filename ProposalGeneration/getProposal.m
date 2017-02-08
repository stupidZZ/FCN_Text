function proposal = getProposal(img, region, regionPerim, clusterCompInfos, estimated_orientation)
   
    %% collecting box and its points
    boxesPoints = zeros(length(clusterCompInfos) * 4, 2);
    boxes = zeros(length(clusterCompInfos), 4);
    for nComp = 1 : length(clusterCompInfos)
        boxes(nComp, :) = clusterCompInfos{nComp}.box;
        if(size(clusterCompInfos{nComp}.box8d, 2) == 8)
            boxesPoints((nComp - 1) * 4 + 1 : (nComp - 1) * 4 + 4, 1) = clusterCompInfos{nComp}.box8d(2 : 2 : 8);
            boxesPoints((nComp - 1) * 4 + 1 : (nComp - 1) * 4 + 4, 2) = clusterCompInfos{nComp}.box8d(1 : 2 : 8);
        else
            boxesPoints((nComp - 1) * 4 + 1 : (nComp - 1) * 4 + 4, 1) = clusterCompInfos{nComp}.box8d(:, 1);
            boxesPoints((nComp - 1) * 4 + 1 : (nComp - 1) * 4 + 4, 2) = clusterCompInfos{nComp}.box8d(:, 2);
        end
%         boxesPoints((nComp - 1) * 4 + 1, :) = [boxes(nComp, 2), boxes(nComp, 1)];
%         boxesPoints((nComp - 1) * 4 + 2, :) = [boxes(nComp, 2) + boxes(nComp, 4) - 1, boxes(nComp,1)];
%         boxesPoints((nComp - 1) * 4 + 3, :) = [boxes(nComp, 2) + boxes(nComp, 4) - 1, ...
%                                                 boxes(nComp,1) + boxes(nComp,3) - 1];
%         boxesPoints((nComp - 1) * 4 + 4, :) = [boxes(nComp, 2), boxes(nComp,1) + boxes(nComp, 3) - 1];
    end
    
    center_x = mean(boxes(:,1) + 0.5 * boxes(:,3));
    center_y = mean(boxes(:,2) + 0.5 * boxes(:,4));
    
    orientationDiff = computeOrientationDiff(regionPerim, [center_y, center_x], estimated_orientation);
    
    boundary_pixels = regionPerim(orientationDiff < 3, :);
    
    %[rectX, rectY, area, ~, sideLenght] = computeMergedBox(boxes, boundary_pixels);
    [rectX, rectY, area, ~, sideLength] = computeMergedBoxByOrientation(img, cat(1, boxesPoints, boundary_pixels), estimated_orientation);
    [width, height] = findWidthAndHeight(rectX, rectY, sideLength, estimated_orientation);

    proposal = zeros(1, 8);
    proposal(1 : 2 : 8) = rectX(1 : 4);
    proposal(2 : 2 : 8) = rectY(1 : 4);
end

