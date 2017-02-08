function [ ret_box8d ] = getRotateBox8D( box8d, orientation, center_x, center_y)
    boxes = box8d;
    rotate_mat = [cosd(orientation), -sind(orientation); sind(orientation), cosd(orientation)];

    %% move to center
    boxes(:,1) = boxes(:,1) - center_y;
    boxes(:,2) = boxes(:,2) - center_x;
    boxes = boxes(:,[2 1])';
    ret_box8d = rotate_mat * boxes;
    ret_box8d = ret_box8d([2,1],:)';
    
    %% resotre position
    ret_box8d(:,1) = ret_box8d(:,1) + center_y;
    ret_box8d(:,2) = ret_box8d(:,2) + center_x;
    
    ret_box8d = round(ret_box8d);
end

