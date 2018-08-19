# MatConvNet tutorial：Train your own data
##  用[MatConvNet](https://github.com/vlfeat/matconvnet)训练自己的数据

## 安装和编译MatConvNet(Build the [library](https://github.com/vlfeat/matconvnet) with CUDA)

	git clone https://github.com/vlfeat/matconvnet
	cd matconvnet
	%create a new file called compileGPU.m and save its contents as:
	addpath matlab
	vl_compilenn('enableGpu', true, ...
               'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0', ...
               'cudaMethod', 'nvcc');%,...
	%                'enableCudnn', 'true',...
	%                'cudnnRoot','E:\MachineLearning\DeepLearning\CuDNN\CUDNNv4') ;
	%

	%then setup the mex environment
	%please select VS2015 or greater
	mex -setup c
	mex -setup cpp
	%finally compile it
	compileGPU

## 准备数据Prepare data

在这里从EasyPR获取了车牌数据(解压[data.zip](data.zip)即可),0-9共10类字符,每类字符存放在一个子文件夹下,如下图所示：

![](https://i.imgur.com/j3zJ0YL.jpg)

代码加载数据的部分位于cnn_plate_setup_data.m，请自行调节输入图像大小

	inputSize =[20,20,1];

数据存放的路径在startup.m

	datadir='data';

## 编写网络结构Setup the net structure

参考cnn_plate_init.m编写网络结构，构建了3层卷积和池化的网络，激活函数为ReLU.

	f=1/100 ;
	net.layers = {};
	net.layers{end+1} = struct('type', 'conv', ...
	                           'weights', {{f*randn(3,3,1,20, 'single'), zeros(1, 20, 'single')}}, ...
	                           'stride', 1, ...
	                           'pad', 0) ;
	net.layers{end+1} = struct('type', 'pool', ...
	                           'method', 'max', ...
	                           'pool', [2 2], ...
	                           'stride', 2, ...
	                           'pad', 0) ;
	net.layers{end+1} = struct('type', 'relu') ;
	net.layers{end+1} = struct('type', 'conv', ...
	                           'weights', {{f*randn(3,3,20,100, 'single'),zeros(1,100,'single')}}, ...
	                           'stride', 1, ...
	                           'pad', 0) ;
	net.layers{end+1} = struct('type', 'pool', ...
	                           'method', 'max', ...
	                           'pool', [2 2], ...
	                           'stride', 2, ...
	                           'pad', 0) ;
	net.layers{end+1} = struct('type', 'relu') ;
	net.layers{end+1} = struct('type', 'conv', ...
	   'weights', {{f*randn(3,3,100,65, 'single'),zeros(1,65,'single')}}, ...
	   'stride', 1, ...
	   'pad', 0) ;
	net.layers{end+1} = struct('type', 'softmaxloss') ;
	
	% Meta parameters
	net.meta.inputSize = [20 20 1] ;
	net.meta.trainOpts.learningRate = logspace(-3, -5, 100);
	net.meta.trainOpts.numEpochs = 50 ;
	net.meta.trainOpts.batchSize = 1000 ;
	
	% Fill in defaul values
	net = vl_simplenn_tidy(net) ;

## 训练Train

运行cnn_plate.m训练网络,训练过程中的曲线如下图所示,可以看出很快就到达99%的准确率.

![](https://i.imgur.com/4MFOZY8.jpg)

## 测试Demo

demo.m展示了如何使用训练好的模型

![](https://i.imgur.com/iaDjqV1.jpg)


Note:记得修改netpath为自己训练的模型哟.

## 参考Reference

[caffe一键式集成开发环境](https://github.com/imistyrain/caffe-oneclick)

[mxnet训练自己的数据](https://github.com/imistyrain/mxnet-oneclick)