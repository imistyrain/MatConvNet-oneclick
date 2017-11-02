netpath=[opts.expDir '/net-epoch-50.mat'];
if not (exist(netpath))
    cnn_plate;
end
class=1;index=1;
subdir=dir(datadir);
imgfiles=dir(fullfile(datadir,subdir(class+2).name));
img=imread(fullfile(datadir,subdir(class+2).name,imgfiles(index+2).name));
imshow(img);
net=load([opts.expDir '/net-epoch-50.mat']);
net=net.net;
im_=single(img);
im_=imresize(im_,net.meta.inputSize(1:2));
im_=im_ - net.meta.normalization.averageImage;
opts.batchNormalization = false ;
net.layers{end}.type = 'softmax';
res=vl_simplenn(net,im_);
scores=squeeze(gather(res(end).x));
[bestScore,best]=max(scores);
str=[subdir(best+2).name ':' num2str(bestScore)];
title(str);
disp(str);