function orientationDiff = computeOrientationDiff(pixels, relativePixel, estimated_orientation)
    diffVector = zeros(size(pixels,1), 2);
    diffVector(:, 1) = pixels(:, 1) - relativePixel(1);
    diffVector(:, 2) = pixels(:, 2) - relativePixel(2);
    
    atanDiff = - atan(diffVector(: ,1) ./ (diffVector(:, 2) + eps)) * 180 / pi;
    
    orientationDiff = abs(atanDiff - estimated_orientation);
    orientationDiff = min(180 - orientationDiff, orientationDiff);
end
