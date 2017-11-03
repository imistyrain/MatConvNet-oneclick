# MatConvNet tutorial
##  用[MatConvNet](https://github.com/vlfeat/matconvnet)训练自己的数据

## 安装和编译MatConvNet

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
	%finally you got it
	compileGPU

## 准备数据

在这里从EasyPR获取了车牌数据(解压[data.zip](data.zip)即可),0-9共10类字符,每类字符存放在一个子文件夹下,如下图所示：

![](https://i.imgur.com/j3zJ0YL.jpg)
## 编写网络结构

参考cnn_plate_init.m编写网络结构

## 训练

运行cnn_plate.m训练网络

## 测试

demo.m展示了如何使用训练好的模型

## 参考

[caffe一键式集成开发环境](https://github.com/imistyrain/caffe-oneclick)

[mxnet训练自己的数据](https://github.com/imistyrain/mxnet-mr)