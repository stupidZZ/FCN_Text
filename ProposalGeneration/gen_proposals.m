function proposals = gen_proposals(img, map, resizeRatio, minmaxRF)
    global param
    global globalVar
    
    %% bw prob map
    f = fspecial('gaussian',[5 5],7);
    map = imfilter(map,f,'same');
    
    bwmap = map > param.minRegionProb;
    
    SE1=strel('disk',3);
    bwmap = imerode(bwmap,SE1);
    bwmap = imdilate(bwmap,SE1);
    bwmap = bwmap>0;
    bwmap = imfill(bwmap,'holes');

    
    %% process each region
    [L, L_num] = bwlabel(bwmap, 8);
    proposals = cell(L_num, 1);
    estimated_orientations = zeros(L_num, 1);
    regions = cell(L_num, 1);
    regionPerims = cell(L_num, 1);
    totalTimeOfGetCompOfRegion = 0;
    totalTimeOfEstimateOrientation = 0;
    totalTimeOfGenProposalsByOrientation = 0;
    for i = 1 : L_num
        regionMap = L == i;
        
        %% construct regions
        [y, x] = find(regionMap);
        regions{i} = [y, x];
        
        %% Substruct sub img and sub region map by regions
        regionMinX= min(x);
        regionMaxX = max(x);
        regionMinY = min(y);
        regionMaxY = max(y);
        regionHeight = regionMaxY - regionMinY + 1;
        regionWidth = regionMaxX - regionMinX + 1;
        
        extendRegionMinX = max(1, regionMinX - regionWidth * 0.005);
        extendRegionMaxX = min(size(img, 2), regionMaxX + regionWidth * 0.005);
        extendRegionMinY = max(1, regionMinY - regionHeight * 0.005);
        extendRegionMaxY = min(size(img, 1), regionMaxY + regionHeight * 0.005);
        
        subImg = img(extendRegionMinY : extendRegionMaxY, extendRegionMinX : extendRegionMaxX, :);
        subRegionMap = regionMap(extendRegionMinY : extendRegionMaxY, extendRegionMinX : extendRegionMaxX);
        [subY, subX] = find(subRegionMap);
        subRegions = [subY, subX];
        
        %% Get comp info from subImg
        %comp_infos = normal_mser3(subImg, param.mser_info, resizeRatio, minmaxRF);
        comp_infos = normal_mser(subImg, resizeRatio, minmaxRF, [globalVar.imgName, '_', num2str(i)]);
        
        %% ---- for debug ----
        if(param.debug)
            boxes = zeros(length(comp_infos), 4);
            for n = 1 : length(comp_infos)
                boxes(n,:) = comp_infos{n}.box; 
            end
            show_bbox(subImg, boxes);
        end
        
        %% construct regionPerims
        subRegionMap = imfill(subRegionMap, 'holes');
        perimMap = bwperim(subRegionMap);
        [y, x] = find(perimMap);
        subRegionPerims = [y, x];
        
        tic;
        [region_comp_infos] = getCompOfRegion(subImg, subRegions, comp_infos, param.minRegionCompCoveredArea, param.maxRegionCompArea);
        tmpTime = toc;
        totalTimeOfGetCompOfRegion = totalTimeOfGetCompOfRegion + tmpTime;
        
        %% generate secondary_region_comp_infos
%         [~, secondary_region_comp_idx] = getCompOfRegion(img, regions{i}, comp_infos, param.secondaryMinRegionCompArea);
%         secondary_region_comp_flags = false(length(comp_infos), 1);
%         secondary_region_comp_flags(secondary_region_comp_idx) = true;
%         secondary_region_comp_flags(region_comp_idx) = false;
%         secondary_region_comp_infos = num2cell(comp_infos(secondary_region_comp_flags));
        
        if(param.debug)
            debug_bbox = zeros(length(region_comp_infos), 4);
            for d_i = 1 : length(region_comp_infos)
                debug_bbox(d_i,:) = region_comp_infos{d_i}.box;
            end
            show_bbox(subImg, debug_bbox);
        end
        
        if(false && param.debug)
            debug_bbox = zeros(length(secondary_region_comp_infos), 4);
            for d_i = 1 : length(secondary_region_comp_infos)
                debug_bbox(d_i,:) = secondary_region_comp_infos{d_i}.box;
            end
            show_bbox(subImg, debug_bbox);
        end
        
        tic;
        estimated_orientations(i) = estimateOrientation(subImg, subRegions, region_comp_infos, param.orient_param);
        tmpTime = toc;
        totalTimeOfEstimateOrientation = totalTimeOfEstimateOrientation + tmpTime;
        
        tic;
        proposals{i} = genProposalsByOrientation(subImg, subRegions, subRegionPerims, estimated_orientations(i), region_comp_infos);
        tmpTime = toc;
        totalTimeOfGenProposalsByOrientation = totalTimeOfGenProposalsByOrientation + tmpTime;
        
        proposals{i}(:, 1 : 2 : 8) = proposals{i}(:, 1 : 2 : 8) + extendRegionMinX;
        proposals{i}(:, 2 : 2 : 8) = proposals{i}(:, 2 : 2 : 8) + extendRegionMinY;
    end
    
    if(false)
        fprintf('TotalTime: %.2f, %.2f, %.2f\n', ...
            totalTimeOfGetCompOfRegion, ... 
            totalTimeOfEstimateOrientation, ... 
            totalTimeOfGenProposalsByOrientation);
    end
    
    proposals = cell2mat(proposals);
end

