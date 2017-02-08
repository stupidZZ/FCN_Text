# FCN_Text

Code written by Zheng Zhang(macaroniz1990@gmail.com)

##########################################################################################
1.Introduction.

This project includes the source code and trained model about the text region fcn and proposal generation. We also provide the probability text region maps for ICDAR2015, ICDAR2013 and MSRA-TD500. If you use the resources of this project, please considering cite the paper:

Zhang, Zheng, et al. "Multi-oriented text detection with fully convolutional networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

##########################################################################################

2.Installation

Dependencies:

Proposal Generation: Matlab, Python, OpenCV(before 3.0 version), VL_feat, Pitor dollar toolbox.
For caffe version of TextRegionFCN, please install HED(https://github.com/s9xie/hed) at first.
For torch version of TextRegionFCN, please install torch at first.

##########################################################################################

3.Resources

We provide the predicted text region by our torch model. You can find them in '/data/ICDAR13/', '/data/ICDAR15/' and '/data/MSRA/'.

##########################################################################################

4. Others
The proposal generation in our paper is partially depend on C++ implementation. In this version, we rewrite the code by python and matlab for reading convinent. But the speed of proposal generation of this version is much slower than papers one about 7x due to language feature and the communication cost between python script and matlab.
