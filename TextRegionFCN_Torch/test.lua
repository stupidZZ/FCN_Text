require 'nn'
require 'cunn'
require 'nngraph'
require 'torch'
require 'cutorch'
require 'Utilities'
require 'Training'
require 'Crop'
require 'create_model'
require 'TableCopy'
require 'TableSelect'
require 'optimizer'
require 'image'
require 'mattorch'
torch.setnumthreads(2)
--cutorch.setDevice(2)

local settings     =  {
	nClasses         = 2,  
	model_dir  		     = './model/model_final.t7',
	-- img_dir = {'/home/mkyang/dataset/ICDAR2013/Challenge2_Test_Task12_Images/', '/home/mkyang/dataset/ICDAR2015/ch4_test_images/', '/home/mkyang/dataset/MSRA/im/'},
	-- res_dir              = {'./results/icdar13/', './results/icdar15/', './results/msra/'},
	-- img_dir = {'/home/mkyang/dataset/ICDAR2015/ch4_test_images/', '/home/mkyang/dataset/MSRA/im/'},
	-- res_dir              = {'./results/icdar15/', './results/msra/'},
	img_dir = {'/share/datasets/SceneText/scene_text/MSRA-TD500/test/'},
	res_dir              = {'./results/msra_test/'},
	model_config   =  {
		input_size        = {1,3,500,500},--{1,3,384,512},--batch format
		load_weight_flag  = false,
		useGPU			  = 'gpu',  -- or 'cpu'
		train_test_flag   = 'test',-- or 'test'
		class_criterion   = 'BCECriterion',
	},
	data_info 	= {
		mean_data_dir  = './data/ilsvrc_2012_mean.t7',--./data/synth_mean.t7',
		batchSize	   = 1
	},
	heights = {200,500,1000}--,1200} -- {200,400,800} --{360,720,1080}
}
local mean_data = torch.load(settings.data_info.mean_data_dir);

function list_img(path)
	--print(path)
	local i,t,popen = 0,{},io.popen
	for file_name in popen('ls -a ' .. path .. ' |grep JPG'):lines() do
		i = i + 1
		t[i] = file_name
	end
	return t
end

function whitening(data)
    -- can be changed, given by the detail
    if data:dim() ~= 4 then
        print('data muset be 4D')
    end
    data =data:float()
    local mean = image.scale(mean_data, data:size(4),data:size(3))
    mean = mean:view(1,data:size(2),data:size(3),data:size(4))   
    local data_tmp = torch.FloatTensor(data:size())
    for ii=1,3 do
        data_tmp[{{},ii,{},{}}]:copy(data[{{},3-ii+1,{},{}}])
    end
    data:copy(data_tmp)
    for ii=1,data:size(1) do
        data[{ii,{},{},{}}]:add(-1,mean)
    end
    return data
end

-- create model --
model = create_vgg_model(settings.model_config)
local trained_pas = torch.load(settings.model_dir)
local pas,gradPas=model:parameters()
for ii=1,#pas do
	pas[ii]:copy(trained_pas[ii])
end
model:evaluate()
---- load test dataset ----
for i=1,table.getn(settings.img_dir) do
	
	local img_dir = settings.img_dir[i]       --msra_dir
	local res_dir = settings.res_dir[i]   --msra_res_dir
	print('begin evaluate on ' .. img_dir)
	local img_names = list_img(img_dir)
	local nImg = table.getn(img_names)
	
	print(nImg)
	for ii=1,nImg do
		local timer = torch.Timer()
	
		local src=image.load(img_dir .. img_names[ii] )
		local src_h,src_w = src:size(2),src:size(3)
		for jj=1,table.getn(settings.heights) do
			if src_h>src_w then
				h = settings.heights[jj]
				w = math.floor(h*src_w/src_h)
			else
				w = settings.heights[jj]
				h = math.floor(w*src_h/src_w)
			end
	               
			local img = image.scale(src,w,h)
			-- preparation
			local input = img:clone():mul(255):view(1,3,h,w)
			input = whitening(input):double()
			-- forward
			local output = model:forward(input:cuda()):clone():float():view(1,input:size(3),input:size(4))
			-- save
			local save_path = res_dir .. string.sub(img_names[ii],1,-5) .. '_' .. jj .. '.jpg'
			image.save(save_path,output)				
		end
		-- print(timer:time())
		print(ii .. '/' .. nImg)
	end
	print('over!')
end
	

