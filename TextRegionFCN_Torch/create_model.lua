function splitter(config)
	local conv,deconv = nn.Sequential(),nn.Sequential()
	--print(config)
	for ii=1,#config do
		local conf = config[ii]
		--print(conf)  -- fixed can add convolution
		if conf[1] == 'conv' then
			conv:add(nn.SpatialConvolutionMM(conf[2][1],conf[2][2],conf[2][3],conf[2][4],conf[2][5],conf[2][6],conf[2][7],conf[2][8]))
		elseif conf[1]=='relu' then
			conv:add(nn.ReLU(true))
		elseif conf[1]=='pool' then
			conv:add(nn.SpatialMaxPooling(conf[2][1],conf[2][2],conf[2][3],conf[2][4]))
		elseif conf[1]=='split' then
			if conf[2] == nil then
				deconv = nil
			else
				deconv:add(nn.SpatialConvolutionMM(conf[2][1],conf[2][2],conf[2][3],conf[2][4],conf[2][5],conf[2][6],conf[2][7],conf[2][8]))
			end
		end		
	end
	if deconv == nil then
		return conv
	else
		return conv,deconv
	end
end
function upSampling(config)
	local up_config,table_config = config[1],config[2]
	local upSample = nn.Sequential()
	upSample:add(nn.TableSelect(table_config))
	-- nn.gModule()
	local input_1 = nn.Identity()()
	local input_2 = nn.Identity()()
	if #up_config == 1 then
		local crop = nn.Crop()({input_1,input_2}):annotate{up_config[1][2]}
		local up = nn.gModule({input_1,input_2},{crop})		
		upSample:add(up)
		return upSample
	else  -- add more one convolutional layer --nn.SpatialDeconvolution
		local deconv = nn.SpatialFullConvolution(up_config[1][2][1],up_config[1][2][2],up_config[1][2][3],up_config[1][2][4],up_config[1][2][5],up_config[1][2][6])(input_2):annotate{up_config[1][3]}
		local crop = nn.Crop()({input_1,deconv}):annotate{up_config[2][2]}
		local up = nn.gModule({input_1,input_2},{crop})
		upSample:add(up)
		return upSample
	end
end

function create_vgg_model(config)
	--local vgg_weights = torch.load(config.vgg_16_weight_dir)
	local vgg_weights = torch.Tensor()
	if config.load_weight_flag then
		vgg_weights = torch.load(config.vgg_16_weight_dir)
	end
	local model,model_concat = nn.Sequential(), nn.ConcatTable()	
	---- pahse-1 convolution step ----
	-- split 1
	local config_1 = {{'conv',{3,64,3,3,1,1,35,35}},{'relu'},{'conv',{64,64,3,3,1,1,1,1}},{'relu'},{'split',{64, 1, 1, 1, 1, 1, 0, 0}}}
	local conv1,deconv1=splitter(config_1)
	-- convert parameters
	if config.load_weight_flag then
		local pas,gradPas = conv1:parameters()
		for ii=1,2 do
			local wb = vgg_weights[ii]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	-- split 2
	local config_2 = {{'pool',{2,2,2,2}},{'conv',{64, 128, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{128, 128, 3, 3, 1, 1, 1, 1}},{'relu'},{'split',{128, 1, 1, 1, 1, 1, 0, 0}}}
	local conv2,deconv2=splitter(config_2)
	-- convert parameters
	if config.load_weight_flag then
		local pas,gradPas = conv2:parameters()
		for ii=1,2 do
			local wb = vgg_weights[ii+2]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	-- split 3
	local config_3 = {{'pool',{2,2,2,2}},{'conv',{128, 256, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{256, 256, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{256, 256, 3, 3, 1, 1, 1, 1}},{'relu'},{'split',{256, 1, 1, 1, 1, 1, 0, 0}}}
	local conv3,deconv3=splitter(config_3)
	-- convert parameters
	if config.load_weight_flag then
		local pas,gradPas = conv3:parameters()
		for ii=1,3 do
			local wb = vgg_weights[ii+4]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	-- split 4
	local config_4 = {{'pool',{2,2,2,2}},{'conv',{256, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'split',{512, 1, 1, 1, 1, 1, 0, 0}}}
	local conv4,deconv4=splitter(config_4)
	-- convert parameters
	if config.load_weight_flag then
		local pas,gradPas = conv4:parameters()
		for ii=1,3 do
			local wb = vgg_weights[ii+7]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	-- split 5
	local config_5 = {{'pool',{2,2,2,2}},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 1, 1, 1, 1, 1, 0, 0}}}
	local conv5=splitter(config_5)--deconv5 is nil
	if config.load_weight_flag then
		local pas,gradPas = conv5:parameters()
		for ii=1,3 do
			local wb = vgg_weights[ii+10]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	---- phase-2 concat step ----
	local split1,split2,split3,split4 =nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable()
	split4:add(conv5)
	split4:add(deconv4)
	conv4:add(split4)
	--
	split3:add(conv4)
	split3:add(deconv3)
	conv3:add(split3)
	-- 
	split2:add(conv3)
	split2:add(deconv2)
	conv2:add(split2)
	-- 
	split1:add(conv2)
	split1:add(deconv1)
	conv1:add(split1)
	---- phase-3 add bottom_data split ----
	local bottom_data = nn.Identity()
	model_concat:add(conv1)
	model_concat:add(bottom_data)
	model:add(model_concat)
	model:add(nn.FlattenTable()) --conv3,conv2,conv1,bottom_data
	if config.useGPU == 'gpu' then
		model=model:cuda()
		model:add(nn.TableCopy('torch.CudaTensor','torch.DoubleTensor',6))--nTable is equal to 6
	end
	--return model,nil
	---- phase-4 upsampling step ----
	local config_up1 = {{{'crop','crop_1'}},{6,5}}
	local upSampling1 = upSampling(config_up1)
	local config_up2 = {{{'deconv',{1,1,4,4,2,2},'deconv_2'},{'crop','crop_2'}},{6,4}}
	local upSampling2 = upSampling(config_up2)
	local config_up3 = {{{'deconv',{1,1,8,8,4,4},'deconv_3'},{'crop','crop_3'}},{6,3}}
	local upSampling3 = upSampling(config_up3)
	local config_up4 = {{{'deconv',{1,1,16,16,8,8},'deconv_4'},{'crop','crop_4'}},{6,2}}
	local upSampling4 = upSampling(config_up4)
	local config_up5 = {{{'deconv',{1,1,32,32,16,16},'deconv_5'},{'crop','crop_5'}},{6,1}}
	local upSampling5 = upSampling(config_up5)
	local deconv = nn.ConcatTable()
	deconv:add(upSampling1)
	deconv:add(upSampling2)
	deconv:add(upSampling3)
	deconv:add(upSampling4)
	deconv:add(upSampling5)
	--return model,deconv
	
	---- phase-5 fuse all deconv splits: initial maybe 1/5 ----
	model:add(deconv)
	model:add(nn.JoinTable(2))
	local fuse = nn.SpatialConvolutionMM(5,1,1,1,1,1,0,0)
	local fuse_pas,fuse_pas_grad = fuse:parameters()
	fuse_pas[1]:fill(0.2)
	fuse_pas[2]:fill(0)
	model:add(fuse)
	--model:add(nn.SpatialConvolutionMM(5,1,1,1,1,1,0,0))
	model:add(nn.Sigmoid())

	---- phase-6 if train, add return criterion
	if config.train_test_flag == 'train' then
		model:add(nn.Reshape(config.input_size[3]*config.input_size[4]))		
		---- default loss function BCECriterion()
		local criterion = nn.BCECriterion()
		if config.class_criterion ~= 'BCECriterion' then
			 error('the loss function must be BCECriterion!')
		end
		return model,criterion
	else -- test
		return model
	end
	
end

function create_vgg_model_small(config)
	--load weights
	local vgg_weights = torch.load(config.vgg_16_weight_dir)

	local model,model_concat = nn.Sequential(), nn.ConcatTable()	
	---- pahse-1 convolution step ----
	-- split 1
	local config_1 = {{'conv',{3,64,3,3,1,1,35,35}},{'relu'},{'conv',{64,64,3,3,1,1,1,1}},{'relu'},{'split',{64, 1, 1, 1, 1, 1, 0, 0}}}
	local conv1,deconv1=splitter(config_1)
	if config.load_weight_flag then
		local pas,gradPas = conv1:parameters()
		for ii=1,2 do
			local wb = vgg_weights[ii]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	--return conv1,deconv1
	-- split 2
	local config_2 = {{'pool',{2,2,2,2}},{'conv',{64, 128, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{128, 128, 3, 3, 1, 1, 1, 1}},{'relu'},{'split',{128, 1, 1, 1, 1, 1, 0, 0}}}
	local conv2,deconv2=splitter(config_2)
	if config.load_weight_flag then
		local pas,gradPas = conv2:parameters()
		for ii=1,2 do
			local wb = vgg_weights[ii+2]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	-- split 3
	local config_3 = {{'pool',{2,2,2,2}},{'conv',{128, 256, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{256, 256, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{256, 256, 3, 3, 1, 1, 1, 1}},{'relu'},{'split',{256, 1, 1, 1, 1, 1, 0, 0}}}
	local conv3,deconv3=splitter(config_3)
	if config.load_weight_flag then
		local pas,gradPas = conv3:parameters()
		for ii=1,3 do
			local wb = vgg_weights[ii+4]
			local w,b = wb.w,wb.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	-- split 4
	local config_4 = {{'pool',{2,2,2,2}},{'conv',{256, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 512, 3, 3, 1, 1, 1, 1}},{'relu'},{'conv',{512, 1, 1, 1, 1, 1, 0, 0}}}
	local conv4=splitter(config_4)
	if config.load_weight_flag then
		local pas,gradPas = conv4:parameters()
		for ii=1,3 do
			local wb = vgg_weights[ii+7]
			local w,b = wb.w.w,wb.b.b
			pas[(ii-1)*2+1]:copy(w)
			pas[(ii-1)*2+2]:copy(b)
		end
	end
	---- phase-2 concat step ----
	local split1,split2,split3 =nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable()
	split3:add(conv4)
	split3:add(deconv3)
	conv3:add(split3)
	-- 
	split2:add(conv3)
	split2:add(deconv2)
	conv2:add(split2)
	-- 
	split1:add(conv2)
	split1:add(deconv1)
	conv1:add(split1)
	---- phase-3 add bottom_data split ----
	local bottom_data = nn.Identity()
	model_concat:add(conv1)
	model_concat:add(bottom_data)
	model:add(model_concat)
	model:add(nn.FlattenTable()) --conv3,conv2,conv1,bottom_data
	if config.useGPU == 'gpu' then
		model=model:cuda()
		model:add(nn.TableCopy('torch.CudaTensor','torch.DoubleTensor',5))--nTable is equal to 6
	end
	--return model,nil
	---- phase-4 upsampling step ----
	local config_up1 = {{{'crop','crop_1'}},{5,4}}
	local upSampling1 = upSampling(config_up1)
	local config_up2 = {{{'deconv',{1,1,4,4,2,2},'deconv_2'},{'crop','crop_2'}},{5,3}}
	local upSampling2 = upSampling(config_up2)
	local config_up3 = {{{'deconv',{1,1,8,8,4,4},'deconv_3'},{'crop','crop_3'}},{5,2}}
	local upSampling3 = upSampling(config_up3)
	local config_up4 = {{{'deconv',{1,1,16,16,8,8},'deconv_4'},{'crop','crop_4'}},{5,1}}
	local upSampling4 = upSampling(config_up4)
	--local config_up5 = {{{'deconv',{1,1,32,32,16,16},'deconv_5'},{'crop','crop_5'}},{6,1}}
	--local upSampling5 = upSampling(config_up5)
	local deconv = nn.ConcatTable()
	deconv:add(upSampling1)
	deconv:add(upSampling2)
	deconv:add(upSampling3)
	deconv:add(upSampling4)
	--return model,deconv
	
	---- phase-5 fuse all deconv splits ----
	model:add(deconv)
	model:add(nn.JoinTable(2))
	model:add(nn.SpatialConvolutionMM(4,1,1,1,1,1,0,0))
	model:add(nn.Sigmoid())

	---- phase-6 if train, add return criterion
	if config.train_test_flag == 'train' then
		model:add(nn.Reshape(config.input_size[3]*config.input_size[4]))		
		---- default loss function BCECriterion()
		local criterion = nn.BCECriterion()
		if config.class_criterion ~= 'BCECriterion' then
			 error('the loss function must be BCECriterion!')
		end
		return model,criterion
	else -- test
		return model
	end
	
end



