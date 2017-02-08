function [width, height] = findWidthAndHeight(rectX, rectY, sideLenght, estimated_orientation)
    pixels = zeros(4, 2);
    for n = 1 : 4
        pixels(n, :) = [(rectY(n) + rectY(n+1))/2, (rectX(n) + rectX(n+1))/2];
    end
    orientationDiff1 = computeOrientationDiff(pixels(1, :), pixels(3, :), estimated_orientation);
    orientationDiff2 = computeOrientationDiff(pixels(2, :), pixels(4, :), estimated_orientation);
    if(orientationDiff1 > orientationDiff2)
        width = round(0.5 * (sideLenght(1) + sideLenght(3)));
        height = round(0.5 * (sideLenght(2) + sideLenght(4)));
    else
        width = round(0.5 * (sideLenght(2) + sideLenght(4)));
        height = round(0.5 * (sideLenght(1) + sideLenght(3)));
    end
end
