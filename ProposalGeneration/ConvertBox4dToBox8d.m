function ret_box = ConvertBox4dToBox8d( box )
    x1 = box(1);
    y1 = box(2);
    x2 = box(1) + box(3) - 1;
    y2 = box(2);
    x3 = box(1) + box(3) - 1;
    y3 = box(2) + box(4) - 1;
    x4 = box(1);
    y4 = box(2) + box(4) - 1;
    ret_box = [y1, x1; y2, x2; y3, x3; y4, x4];
end

