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
--torch.setdefaulttensortype('torch.FloatTensor')
setupLogger('log.txt')  -- record logs
--cutorch.setDevice(1)

local settings     =  {
	nClasses         = 2,  
	--classWeight      = torch.Tensor({1,1}),  
	displayInterval  = 500,
	testInterval     = 50000,--50000,--3000,--1000
	snapShotInterval = 10000,--5000,--20000
	maxIterations    = 2000000,
	maxPatience      = 5,
	nLRChange        = 10e3,
	nBoost           = 5,
	model_config   =  {
		input_size        = {1,3,500,500},--{1,3,384,512},--batch format
		load_weight_flag  = true,
		vgg_16_weight_dir = './data/vgg_weights.t7',
		useGPU			  = 'gpu',  -- or 'cpu'
		train_test_flag   = 'train',-- or 'test'
		class_criterion   = 'BCECriterion',--equal to sigmoid cross entropy loss: E=-1/n*sigma[p*logp+(1-p)log(1-p)]	
	},
	data_info 	= {
		train_data_dir = '../dataset/11_2/trainData.t7',
		train_gt_dir   = '../dataset/11_2/trainLabel.t7',
		test_data_dir  = '../dataset/11_2/testData.t7',
		test_gt_dir    = '../dataset/11_2/testLabel.t7',
		mean_data_dir  = './data/ilsvrc_2012_mean.t7',--./data/synth_mean.t7',
		batchSize	   = 1
	},
	--optimMethod   = optim.adagrad
	optimSettings    = {
	learningRate      = 1e-4,-->13 * 20
        weightDecay       = 5e-4,
        momentum          = 0.9,
        learningRateDecay = 0,        
	},

}
-- create model --
model,criterion = create_vgg_model(settings.model_config)
--model,criterion = create_vgg_model(settings.model_config)
-- load params --
paras_init = torch.load('./model_11_2/model_init.t7')
paras,gradParas=model:parameters()
for ii=1,#paras do
	paras[ii]:copy(paras_init[ii])
end
paras_init = nil
collectgarbage()
---
model:training()
--begin training --
logging('begin to train model ->\n')
model = train_model(model,criterion,settings)
logging('model training over!\n')
