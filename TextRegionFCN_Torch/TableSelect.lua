local TableSelect, parent = torch.class('nn.TableSelect', 'nn.Module')

function TableSelect:__init(index)--{1,2}
   parent.__init(self)
   self.index = index  --also table
   self.gradInput = {}
   self.output = {}
end

function TableSelect:updateOutput(input)
   assert(type(input)=='table','input must be table')
   for ii=1,table.getn(self.index) do
      self.output[ii]=input[self.index[ii]]
   end
   return self.output
end

function TableSelect:updateGradInput(input, gradOutput)
   assert(type(gradOutput)=='table','gradOutput must be table')
   for ii=1,table.getn(input) do -- table value can't be nil, which is dangerous
      self.gradInput[ii]=input[ii]:clone():fill(0)
   end
   for ii=1,table.getn(self.index) do
      if gradOutput[ii]~=nil then  --crop return nil
         self.gradInput[self.index[ii]]:copy(gradOutput[ii])
      end
   end
   return self.gradInput
end

function TableSelect:type(type)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type)
end

function TableSelect:forward(input)
-- rewrite forward
   return self:updateOutput(input)
end

function TableSelect:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end
