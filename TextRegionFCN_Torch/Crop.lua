local Crop, Parent = torch.class('nn.Crop', 'nn.Module')

function Crop:__init()
	Parent.__init(self)
	self.mask = torch.Tensor()
end

function Crop:updateOutput(_input)
-- input_data: bottom data
-- input: current laye's input, format:nBatch x nChanel x H x W or nChannel x H x W, just one scale for per training time
	local input_data = _input[1];
	local input = _input[2];
	--self.output:resizeAs(input_data)  --:copy(input_data)

	if input_data:dim() == 3 then -- one image
		self.output:resize(input:size(1),input_data:size(2),input_data:size(3))  --:copy(input_data)
		self.mask = torch.Tensor(1,4)  -- pad_l, pad_r,pad_t,pad_b
		self.mask[{1,1}]=math.floor((input:size(2) - input_data:size(2))/2)
		self.mask[{1,2}]=(input:size(2)-input_data:size(2)) - self.mask[{1,1}]
		self.mask[{1,3}]=math.floor((input:size(3) - input_data:size(3))/2)
		self.mask[{1,4}]=(input:size(3) - input_data:size(3)) - self.mask[{1,3}]	
		-- update: crop input
		self.output:copy(input[{{},{self.mask[{1,1}]+1,self.mask[{1,1}]+input_data:size(2)},{self.mask[{1,3}]+1,self.mask[{1,3}]+input_data:size(3)}}]);
		
	elseif input_data:dim() == 4 then  -- batch
		self.output:resize(input:size(1),input:size(2),input_data:size(3),input_data:size(4))  --:copy(input_data)
		--self.mask = torch.Tensor(input_data:size(1),4) 
		self.mask = torch.Tensor(1,4)  -- pad_l, pad_r,pad_t,pad_b
		self.mask[{1,1}]=math.floor((input:size(3) - input_data:size(3))/2)
		self.mask[{1,2}]=(input:size(3)-input_data:size(3)) - self.mask[{1,1}]
		self.mask[{1,3}]=math.floor((input:size(4) - input_data:size(4))/2)
		self.mask[{1,4}]=(input:size(4) - input_data:size(4)) - self.mask[{1,3}]
		-- update: crop input
		self.output:copy(input[{{},{},{self.mask[{1,1}]+1,self.mask[{1,1}]+input_data:size(3)},{self.mask[{1,3}]+1,self.mask[{1,3}]+input_data:size(4)}}])
		--self.output:copy(input[{{},{},{self.mask[{1,1}],self.mask[{1,2}]},{self.mask[{1,3}],self.mask[{1,4}]}}])		
	else
		error('<Crop updateOutput> illegal input, must be 3 D or 4 D')
	end
	-- update crop input
	return self.output
end


function Crop:updateGradInput(_input,gradOutput)
	--self.gradInput = torch.Tensor()
	if gradOutput:dim() == 3 then
		self.gradInput:resize(gradOutput:size(1),gradOutput:size(2)+self.mask[{1,1}]+self.mask[{1,2}],gradOutput:size(3)+self.mask[{1,3}]+self.mask[{1,4}])
		self.gradInput:fill(0)
		self.gradInput[{{},{self.mask[{1,1}]+1,self.mask[{1,1}]+gradOutput:size(2)},{self.mask[{1,3}]+1,self.mask[{1,3}]+gradOutput:size(3)}}]:copy(gradOutput)
	elseif gradOutput:dim() == 4 then
		self.gradInput:resize(gradOutput:size(1),gradOutput:size(2),gradOutput:size(3)+self.mask[{1,1}]+self.mask[{1,2}],gradOutput:size(4)+self.mask[{1,3}]+self.mask[{1,4}])
		self.gradInput:fill(0)
		self.gradInput[{{},{},{self.mask[{1,1}]+1,self.mask[{1,1}]+gradOutput:size(3)},{self.mask[{1,3}]+1,self.mask[{1,3}]+gradOutput:size(4)}}]:copy(gradOutput)
	else
		error('<crop updateGradInput> illegal gradOutput, must be 3 D or 4 D')
	end 
	return {nil,self.gradInput}
end

function Crop:accGradParameters(_input, gradOutput, scale)
end


function Crop:forward(_input)
-- rewrite forward
   return self:updateOutput(_input)
end

function Crop:backward(_input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(_input, gradOutput)
   self:accGradParameters(_input, gradOutput, scale)
   return self.gradInput
end


