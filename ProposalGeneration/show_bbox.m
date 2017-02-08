function show_bbox(im, bbox, colors)
    if(nargin == 2)
        imshow(im);
        for n = 1 : size(bbox,1)
            rectangle('position', bbox(n,:), 'edgecolor', rand(3,1));
        end
    end
    
    if(nargin == 3)
        imshow(im);
        for n = 1 : size(bbox,1)
            rectangle('position', bbox(n,:), 'edgecolor', colors(n,:));
        end
    end
end
