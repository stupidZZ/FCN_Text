function generate_mean(trainData)
    local mean_data = torch.Tensor(trainData:size(2),trainData:size(3),trainData:size(4)):fill(0)
    local batchSize = 128
    local batchData = torch.Tensor(batchSize,trainData:size(2),trainData:size(3),trainData:size(4))
    local nBatch = math.floor(trainData:size(1) / batchSize) 
    local mean_weights = torch.Tensor(nBatch):fill(1)
    if (trainData:size(1) % batchSize) == 0 then
        mean_weights:fill(1)
    else
        nBatch = nBatch + 1
        mean_weights = torch.Tensor(nBatch):fill(1)
        mean_weights[nBatch] = (trainData:size(1) % batchSize) / batchSize 
    end
    local mean_datas = torch.Tensor(nBatch,trainData:size(2),trainData:size(3),trainData:size(4)):fill(0)
    local bc = 1
    for ii=1,trainData:size(1),batchSize do
        local idx_end = math.min(ii+batchSize-1,trainData:size(1))
        if(idx_end-ii+1)==batchSize then   
            batchData:copy(trainData[{{ii,idx_end},{},{},{}}])
            local mean_temp = torch.mean(batchData,1)
            mean_datas[{bc,{},{},{}}]:copy(mean_temp)
            bc = bc + 1
        else
            local subBatchData=batchData:narrow(1,1,idx_end-ii+1)
            subBatchData:copy(trainData[{{ii,idx_end},{},{},{}}])
            local mean_temp = torch.mean(subBatchData,1)
            mean_datas[{bc,{},{},{}}]:copy(mean_temp)
        end
    end
    for ii=1,nBatch do
        local weight = mean_weights[ii]
        local mean_tmp = mean_datas[{ii,{},{},{}}]
        mean_tmp = mean_tmp:view(trainData:size(2),trainData:size(3),trainData:size(4))      
        mean_data:add(weight,mean_tmp)
    end
    mean_data:div(torch.sum(mean_weights))
    return mean_data
end

function train_model(model,criterion,settings)
	-- read config
	local dataInfo = settings.data_info
	local inputInfo = settings.model_config.input_size
	local optimState  = settings.optimSettings
	--local optimMethod = optim.adagrad
    local optimMethod = optimizer_adagrad
	local minLoss	  = math.huge
	local maxPatience    = settings.maxPatience
	local minLearningRate = optimState.learningRate/settings.nLRChange
	--local batchSize   = data_info.batchSize
	logging('load dataset ->\n')
	local trainData = torch.load(dataInfo.train_data_dir)
	local trainLabel = torch.load(dataInfo.train_gt_dir)
	local testData = torch.load(dataInfo.test_data_dir)
	local testLabel = torch.load(dataInfo.test_gt_dir)
	local nTrain = trainData:size(1)
	local nTest  = trainData:size(1)
	logging('dataset loaded !\n')
	local dataCuda = torch.CudaTensor(dataInfo.batchSize,inputInfo[2],inputInfo[3],inputInfo[4])
	local labelCuda = torch.DoubleTensor(dataInfo.batchSize,inputInfo[3]*inputInfo[4])
    local parameters,gradParameters = model:parameters()
    -- discriminate file exist, function defined in Utilities.lua
    if (file_exists(dataInfo.mean_data_dir)) then
        mean_data = torch.load(dataInfo.mean_data_dir);
    else
        -- generate mean data
        mean_data = generate_mean(trainData)
        torch.save(dataInfo.mean_data_dir,mean_data)        
    end    
    --local mean_data = torch.load(dataInfo.mean_data_dir);
    mean_data = mean_data:float()
    mean_data = image.scale(mean_data, inputInfo[4], inputInfo[3])
    logging('data mean loaded!')
    --mean_data = image.scale(mean_data, inputInfo[3], inputInfo[4]);
    -- whitening
    function whitening(data)
        -- can be changed, given by the detail
        if data:dim() ~= 4 then
            print('data muset be 4D')
        end
        data =data:float()
        local data_tmp = torch.FloatTensor(data:size())
        for ii=1,3 do
            data_tmp[{{},ii,{},{}}]:copy(data[{{},3-ii+1,{},{}}])
        end
        data:copy(data_tmp)
        for ii=1,data:size(1) do
            data[{ii,{},{},{}}]:add(-1,mean_data)
            --data[{ii,{},{},{}}]:add(data[{ii,{},{},{}}],-1,mean_data)
        end
        return data
    end
	-- get one batch
	function getOneBatch()
		local idx = torch.LongTensor():randperm(nTrain):narrow(1,1,dataInfo.batchSize)

        --print(idx)
		local batchData = torch.FloatTensor(dataInfo.batchSize,inputInfo[2],inputInfo[3],inputInfo[4])
		local batchLabel = torch.FloatTensor(dataInfo.batchSize,labelCuda:size(2))
		for ii = 1,dataInfo.batchSize do
			index = idx[{ii}]
			local img = torch.Tensor(inputInfo[1],inputInfo[2],inputInfo[3],inputInfo[4]):copy(trainData[{index,{},{},{}}])
			-- whitening
			img = whitening(img)
            -- copy data -- copy label
            batchData[{ii,{},{},{}}]:copy(img)
			batchLabel[{ii,{}}]:copy(trainLabel[{index,{}}])
		end
		return batchData,batchLabel
	end
	-- train one batch
	function trainOneBatch(input,target)
		local nFrame = input:size(1)
		dataCuda:copy(input)
		labelCuda:copy(target)
        local feval = function(x)
            if x ~= parameters then
                --parameters:copy(x)
                for ii=1,#x do
                    parameters[ii]:copy(x[ii])
                end                
            end
            --gradParameters:zero()
            for ii=1,#gradParameters do
                gradParameters[ii]:zero()
            end 
            local f = 0           		
            local output = model:forward(dataCuda)
            local f = criterion:forward(output,labelCuda)
            model:backward(dataCuda,criterion:backward(output,labelCuda))
            -- 
            for ii=1,#gradParameters do
                gradParameters[ii]:div(nFrame)
            end
            f = f/nFrame
            return f,gradParameters
        end
        local _,trainloss = optimMethod(feval,parameters,optimState);trainloss = trainloss[1]
        return trainloss
    end
    -- doTest()
    function doTest()
    	local testLoss = 0
    	for t=1,testData:size(1),dataInfo.batchSize do
    		local idx_end = math.min(t+dataInfo.batchSize-1,testData:size(1))
    		if(idx_end-t+1)==dataInfo.batchSize then    	
                local imgs = torch.Tensor(dataInfo.batchSize,inputInfo[2],inputInfo[3],inputInfo[4]):copy(testData[{{t,idx_end},{},{},{}}])
    			imgs = whitening(imgs)
                dataCuda:copy(imgs)
    			labelCuda:copy(testLabel[{{t,idx_end},{}}])
    			local output = model:forward(dataCuda)
    			testLoss = testLoss + criterion:forward(output,labelCuda) 
    		else
    			local subDataCuda = dataCuda:narrow(1,1,idx_end-t+1)
    			local subLabelCuda= labelCuda:narrow(1,1,idx_end-t+1)
                local imgs = torch.Tensor(idx_end-t+1,inputInfo[2],inputInfo[3],inputInfo[4]):copy(testData[{{t,idx_end},{},{},{}}])
    			imgs = whitening(imgs)
                subDataCuda:copy(imgs)
    			subLabelCuda:copy(testLabel[{{t,idx_end},{}}])
    			local output = model:forward(subDataCuda)
    			testLoss = testLoss + criterion:forward(output,subLabelCuda)
    		end
    	end
    	testLoss = testLoss/testData:size(1)
    	logging(string.format('loss on test dataset ->\n%s', testLoss))
    	return testLoss
    end

    -- main for training
 	local iterations = 0
    local loss = 0
    local patience = maxPatience
	while(true)do
		-- train one batch
		local input,target = getOneBatch()
		loss = loss + trainOneBatch(input,target)
		iterations = iterations + 1	
		-- display the loss, at interval
		if iterations % settings.displayInterval == 0 then
            loss = loss / settings.displayInterval
            logging(string.format('Iteration %d - train loss = %f, lr = %f',
                iterations, loss, optimState.learningRate))
            loss = 0
        end
        -- display the testing result
        if iterations % settings.testInterval == 0 then
        	logging('Testing ...')
        	model:evaluate()
        	local testLoss = doTest()
        	logging(string.format('Test ERROR is:%f (bestErr = %f)',testLoss,minLoss))
        	if testLoss < minLoss then
        		minLoss = testLoss
        		patience = maxPatience
        	else
        		patience = patience - 1
        		if patience<0 then
        			optimState.learningRate = 0.1*optimState.learningRate
        			if optimState.learningRate < minLearningRate then
        				logging('Maximum patience reached,terminating')
        				break
        			end
        			patience = maxPatience
        		end
        	end

            model:training()
        end
        -- store the models
		if iterations % settings.snapShotInterval == 0 then
            torch.save(string.format('model_11_2/model_%d.t7', iterations),parameters)
        end
        if iterations >= settings.maxIterations then
            logging('Maximum iterations reached, terminating ...')
            break
        end
    end
	return model
end
