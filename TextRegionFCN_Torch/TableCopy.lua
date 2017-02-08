local TableCopy, parent = torch.class('nn.TableCopy', 'nn.Module')

function TableCopy:__init(intype,outtype,nTable,forceCopy,dontCast)
   intype = intype or torch.Tensor.__typename
   outtype = outtype or torch.Tensor.__typename
   self.nt = nTable
   self.dontCast = dontCast
   parent.__init(self)
   self.gradInput = {}
   self.output = {}
   for ii=1,self.nt do
      self.gradInput[ii]=torch.getmetatable(intype).new()
      self.output[ii]=torch.getmetatable(outtype).new()
   end
   if (not forceCopy) and intype == outtype then
      self.updateOutput = function(self, input)
                        self.output = input
                        return input
                     end

      self.updateGradInput = function(self, input, gradOutput)
                         self.gradInput = gradOutput
                         return gradOutput
                      end
   end

end

function TableCopy:updateOutput(input)
   for ii=1,self.nt do
      self.output[ii]:resize(input[ii]:size()):copy(input[ii])
   end
   return self.output
end

function TableCopy:updateGradInput(input, gradOutput)
   for ii=1,self.nt do 
      self.gradInput[ii]:resize(gradOutput[ii]:size()):copy(gradOutput[ii])
   end
   return self.gradInput
end

function TableCopy:type(type)
   if type and self.dontCast then
      return self
   end
   return parent.type(self, type)
end