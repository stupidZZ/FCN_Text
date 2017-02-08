function [ region_comp, region_comp_idx ] = getCompOfRegion(img, region, comp_infos, minRegionCompCoveredArea, maxRegionCompArea)
    %Get components which inside in the region
    global param;
    
    global getCompOfRegion_px;
    global getCompOfRegion_py;
    
    getCompOfRegion_px = region(:,2);
    getCompOfRegion_py = region(:,1);
        
    if(true && param.debug)
        for i = 1 : size(region, 1)
            img(region(i,1), region(i,2), :) = [1,1,1];
        end
    end
    
    maxRegionCompAreaValue = size(region, 1) * maxRegionCompArea;
    
    region_comp_flag = false(length(comp_infos), 1);    
    for i = 1 : length(comp_infos)
        comp_box = comp_infos{i}.box;
        in_pixel_num = length(find((getCompOfRegion_px >= comp_box(1)) & ...
                                   (getCompOfRegion_px <= comp_box(1) + comp_box(3) - 1) & ...
                                   (getCompOfRegion_py >= comp_box(2)) & ...
                                   (getCompOfRegion_py <= comp_box(2) + comp_box(4) - 1)));
        comp_box_area = comp_box(3) * comp_box(4);
        if(in_pixel_num > comp_box_area * minRegionCompCoveredArea && comp_box_area < maxRegionCompAreaValue)
            region_comp_flag(i) = true;
        end
    end
    region_comp = comp_infos(region_comp_flag);
    region_comp_count = sum(region_comp_flag);
        
    box = zeros(region_comp_count, 4);
    for i = 1 : region_comp_count
        box(i,:) = region_comp{i}.box;
    end
    nms_idx = box_nms(box, box(:,3) .* box(:,4), 0.8);
    
    region_comp = region_comp(nms_idx);
end
