function [ rectX, rectY, area, perimeter, sideLength] = computeMergedBoxByOrientation(img, pixels, estimated_orientation)
    %% Compute convex hull
    DT = delaunayTriangulation(pixels(:, 2), pixels(:, 1));
    convexPointIdx = convexHull(DT);
    convexPoints = DT.Points(convexPointIdx, :);
    convexPoints = convexPoints(:, [2 1]);
%     imshow(img);
%     hold on;
%     plot(convexPoints(:, 2), convexPoints(:, 1));
%     hold off;
    
    %% Compute centralized pixels
    cx = mean(convexPoints(:, 2));
    cy = mean(convexPoints(:, 1));
     
    centralizedPixels = pixels;
    centralizedPixels(:, 1) = centralizedPixels(:, 1) - cy;
    centralizedPixels(:, 2) = centralizedPixels(:, 2) - cx;
    
    %% compute rotate pixels
    rotate_mat = [cosd(estimated_orientation), -sind(estimated_orientation); sind(estimated_orientation), cosd(estimated_orientation)];
    centralizedPixels = centralizedPixels(:, [2, 1]);
    rotatePixels = (rotate_mat * centralizedPixels')';
    
    %% swap X and Y
    rotatePixels = rotatePixels(:, [2, 1]);
    centralizedPixels = centralizedPixels(:, [2, 1]);

    %% compute min_box of rotate pixels
    rotate_min_box = [min(rotatePixels(:,2)), ...
                        min(rotatePixels(:,1)), ...
                        max(rotatePixels(:,2)) - min(rotatePixels(:,2)), ...
                        max(rotatePixels(:,1)) - min(rotatePixels(:,1))];
    rotate_min_box8p = ConvertBox4dToBox8d(rotate_min_box);
    rotate_min_box8p = getRotateBox8D(rotate_min_box8p, -estimated_orientation, 0, 0);
    
    
    
    rectX = [rotate_min_box8p(:, 2); rotate_min_box8p(1, 2)] + cx;
    rectY = [rotate_min_box8p(:, 1); rotate_min_box8p(1, 1)] + cy;
    
    area = polyarea(rectX, rectY);      
    sideLength = ((rectY(2 : 5) - rectY(1 : 4)) .^ 2 + (rectX(2 : 5) - rectX(1 : 4)) .^ 2) .^ 0.5;
    perimeter = sum(sideLength);
end
