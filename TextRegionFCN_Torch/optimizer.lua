function optimizer_adagrad(opfunc, x, config, state)
--opfunc: outputs including loss(fx),grad_loss(df/dx); x:parameters; 
   -- (0) get/update state
   if config == nil and state == nil then
      print('no state table, ADAGRAD initializing')
   end
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
      
   -- (4) parameter update with single or individual learning rates
   if not state.paramVariance then
      state.paramVariance = {}
      state.paramStd = {}
      for ii=1,#x do
         state.paramVariance[ii] = torch.Tensor():typeAs(x[ii]):resizeAs(dfdx[ii]):zero()
         state.paramStd[ii] = torch.Tensor():typeAs(x[ii]):resizeAs(dfdx[ii])   
      end
   end
   for ii=1,#x do
      state.paramVariance[ii]:addcmul(1,dfdx[ii],dfdx[ii])
      state.paramStd[ii]:resizeAs(state.paramVariance[ii]):copy(state.paramVariance[ii]):sqrt()
      if ii<=26 then
         x[ii]:addcdiv(-clr, dfdx[ii],state.paramStd[ii]:add(1e-10))   
      else
         x[ii]:addcdiv(-clr*1e1, dfdx[ii],state.paramStd[ii]:add(1e-10))   
      end
   end

   -- (5) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end