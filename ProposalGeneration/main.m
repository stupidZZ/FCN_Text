function main()
    addpath('../include/vlfeat-0.9.20/toolbox/');
    vl_setup();
    addpath(genpath('../include/eccv14text'));
    %data_infos
    data_infos.img_path = '../data/msra_torch/im/';
    data_infos.map_path = '../data/msra_torch/multiscale/';
    data_infos.res_path = '../data/msra_torch/proposal_res/';
    
    %global variable
    global globalVar
    
    %gen_proposals param
    global param
    param.workPath = '/Users/zhangzheng/Documents/FCN_FULL/genProposal';
    param.reuse_mser = false;
    param.debug = false;
    param.minRegionProb = 0.2;
    param.minRegionCompCoveredArea = 0.7;
    param.maxRegionCompArea = 1;
    param.secondaryMinRegionCompArea = 0.5;
    param.orient_param.orientationInterval = 2;
    param.orient_param.minOrientation = -90;
    param.orient_param.maxOrientation = 90;
    param.minCompHeightSimilarity = 0.7;
    param.maxCompOrientationDiff = 3;
    param.minIoUDiff = 0.85;
    param.maxDistRatio = 2;
    %minmaxRFs = [137, 68, 5];
    minmaxRFs = [137, 32, 5];
    
    %% Only used for normal_mser3
    param.mser_info.delta = 1;
    param.mser_info.minArea = 0.002;
    param.mser_info.maxArea = 1;
    param.mser_info.minDiversity = 0.8;
    param.mser_info.maxVariation = 0.15;

    mkdir(data_infos.res_path);
    
    extension = '.jpg';
    imgData = dir([data_infos.img_path,'*.jpg']);% original image.
    if(length(imgData) == 0)
       imgData = dir([data_infos.img_path,'*.JPG']);% original image.
       extension = '.JPG';
    end
    nImg = length(imgData);
    for ii= 1:nImg
        disp(ii);
        [~, name, ~] = fileparts(imgData(ii).name);
%         if(~strcmp(name, 'img_11'))
%             continue;
%         end

        globalVar.imgName = name;
        img_path = [data_infos.img_path, imgData(ii).name];%the original image.
        
        proposalsSavePath = [data_infos.res_path, name, '.txt'];
        nMap = 3;
        proposals = cell(nMap, 1);
        for jj = 1 : nMap
            map_path = [data_infos.map_path, name, '_', num2str(jj), extension];%the res image from last phase.
            img = imread(img_path);
            map = imread(map_path);
            map = double(map) / 255;
            
            [map_h,map_w,~]=size(map);
            resizeRatio = size(img, 1) / map_h;
            img = imresize(img, [map_h, map_w], 'bilinear');
            
            proposals_tmp = gen_proposals(img, map, resizeRatio, minmaxRFs(jj));
            if(isempty(proposals_tmp) == false)
                proposals_tmp(:, 1 : 8) = proposals_tmp(:, 1 : 8) * resizeRatio;
            end
            proposals{jj} = proposals_tmp;
            
            if(false)
                imshow(img);
                hold on;
                for nProposal = 1 : size(proposals_tmp, 1)
                    x_arr = proposals_tmp(nProposal, 1 : 2 : 8);
                    y_arr = proposals_tmp(nProposal, 2 : 2 : 8);
                    plot([x_arr, x_arr(1)], [y_arr, y_arr(1)], 'color', rand(3,1));
                end
                hold off;
                saveas(gcf, [data_infos.res_path, name, '_', num2str(jj), '.jpg'], 'jpg');
            end
        end
        proposals = cell2mat(proposals);
        
        if(true)
            img = imread(img_path);
            imshow(img);
            hold on;
            for nProposal = 1 : size(proposals, 1)
                x_arr = proposals(nProposal, 1 : 2 : 8);
                y_arr = proposals(nProposal, 2 : 2 : 8);
                plot([x_arr, x_arr(1)], [y_arr, y_arr(1)], 'color', rand(3,1));
            end
            hold off;
            saveas(gcf, [data_infos.res_path, name, '_all.jpg'], 'jpg');
        end
        
        %% to adapter old code
        proposalsToSave = zeros(10, size(proposals, 1));
        if(size(proposalsToSave, 2) ~= 0)
            proposalsToSave(1 : 8, :) = round(proposals(:, 1 : 8))';
            proposalsToSave(10, :) = proposals(:, 9);
        end
        fid = fopen(proposalsSavePath, 'w');
        fprintf(fid, '%d %d %d %d %d %d %d %d %d %f\n', proposalsToSave);
        fclose(fid);
    end
end