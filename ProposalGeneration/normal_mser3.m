function comp_infos = normal_mser3(src,info, resizeRatio, minmaxRF)
%this script use the normal version of mser extract regions
%default: one channel
img = imadjust(rgb2gray(src));
%mser
[bbox,pixelsList,bright_dark_flag] = mser(img,info);

% filter bbox
if(~isempty(bbox))
    rule1 = (bbox(:,3) ./ bbox(:,4)) < 2.5;
    rule2 = (bbox(:,4) ./ bbox(:,3)) < 5;
    rule3 = (max(bbox(:,3), bbox(:,4)) * resizeRatio) > minmaxRF;
    rule4 = bbox(:,3) .* bbox(:,4) > 20;
    rule_Ind = rule1 & rule2 & rule3 & rule4;
    bbox = bbox(rule_Ind,:);
    pixelsList = pixelsList(rule_Ind);
end

%record
comp_infos=cell(size(bbox,1), 1);
for kk=1:size(bbox,1)
    box=bbox(kk,[2 1 4 3]);
    comp_infos{kk}.box=box;
    comp_infos{kk}.center=floor([box(1)+box(3)/2,box(2)+box(4)/2]);
    comp_infos{kk}.pixelList = pixelsList{kk,1};
    comp_infos{kk}.bright_dark_flag = bright_dark_flag(kk);
    comp_infos{kk}.box8d = ConvertBox4dToBox8d(box);
end

%%%%%%%% function %%%%%%%%
function  [bbBox,pixelsList,bright_dark_flag] = mser(I,info)
 %
[h,w]=size(I);
%phase-1: 'BrightOnDark'
[r,f] = vl_mser(I,'MinDiversity',info.minDiversity,'MaxVariation',info.maxVariation,...
				'MaxArea',info.maxArea,'MinArea',info.minArea,'BrightOnDark',1,'DarkOnBright',0,...
				'Delta',info.delta);
M =zeros(size(I));
for x=r'
    s = vl_erfill(I,x);
    M(s) = M(s) + 1;%if s belong to one region, add 1.
end
mCount = max(max(M));
bbBox = [];
pixelsList = {};
bright_dark_flag=[];
nCount = 1;
for ii=mCount:-1:1
    MM = M;
    MM(find(MM<ii))=0;
    MM(find(MM>=ii))=1;
    mContours = bwlabel(MM,8);
    mNum = max(max(mContours));
    for jj=1:mNum
        [idx,idy,~] = find(mContours == jj);    
        ww = max(idx) - min(idx) + 1;%note me: ww & hh are reverse
        hh = max(idy) - min(idy) + 1;
		%condition
		if (ww/(hh+eps)<0.1)||(hh/(ww+eps)<0.1)
			continue;
		end
		if (ww<4||hh<4) || (ww>400|| hh>400)
			continue;
        end	
        if ww*hh< 20
            continue;
        end
		%update
		bbBox(nCount,:)=[min(idx),min(idy),ww,hh];
		pixelsList{nCount,1}=[idx,idy];
        bright_dark_flag(nCount)=1;
		nCount = nCount + 1;
    end
end

%phase-2: 'DarkOnBright'
[r,f] = vl_mser(I,'MinDiversity',info.minDiversity,'MaxVariation',info.maxVariation,...
				'MaxArea',info.maxArea,'MinArea',info.minArea,'BrightOnDark',0,'DarkOnBright',1,...
				'Delta',info.delta);
M =zeros(size(I));
for x=r'
    s = vl_erfill(I,x);
    M(s) = M(s) + 1;%if s belong to one region, add 1.
end
mCount = max(max(M));
for ii=mCount:-1:1
    MM = M;
    MM(find(MM<ii))=0;
    MM(find(MM>=ii))=1;
    mContours = bwlabel(MM,8);
    mNum = max(max(mContours));
    for jj=1:mNum
        [idx,idy,~] = find(mContours == jj);     
        ww = max(idx) - min(idx) + 1;%note me: ww & hh are reverse
        hh = max(idy) - min(idy) + 1;
		%condition
		if (ww/(hh+eps)<0.1)||(hh/(ww+eps)<0.1)
			continue;
		end
		if (ww<4||hh<4) || (ww>400|| hh>400)
			continue;
        end	
        if ww*hh< 20
            continue;
        end
		%update
		bbBox(nCount,:)=[min(idx),min(idy),ww,hh];
		pixelsList{nCount,1}=[idx,idy];
        bright_dark_flag(nCount)=0;
		nCount = nCount + 1;
    end
end
end

end