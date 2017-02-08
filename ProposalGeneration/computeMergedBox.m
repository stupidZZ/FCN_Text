function [ rectX, rectY, area, perimeter, sideLength] = computeMergedBox( boxes, append_pixels )
   nBoxes = size(boxes, 1);
   compPoints = zeros(nBoxes* 4, 2);
   for nComp = 1 : size(boxes,1)
       loopBox = boxes(nComp, :);
       compPoints((nComp - 1) * 4 + 1, :) = [loopBox(2), loopBox(1)];
       compPoints((nComp - 1) * 4 + 2, :) = [loopBox(2), loopBox(1) + loopBox(3) - 1];
       compPoints((nComp - 1) * 4 + 3, :) = [loopBox(2) + loopBox(4) - 1, loopBox(1) + loopBox(3) - 1];
       compPoints((nComp - 1) * 4 + 4, :) = [loopBox(2) + loopBox(4) - 1, loopBox(1)];
   end
   
   if(nargin == 2)
       compPoints = cat(1, compPoints, append_pixels);
   end
   
   [rectX, rectY, area, perimeter] = minboundrect(compPoints(:, 2), compPoints(:, 1));
   
   sideLength = ((rectX(2 : 5) - rectX(1 : 4)) .^ 2 + (rectY(2 : 5) - rectY(1 : 4)).^2).^0.5;
end

