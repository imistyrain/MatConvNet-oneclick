function [net, info] = cnn_plate()
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
%datadir='E:\MachineLearning\caffe\caffe-windows-master\platerecognition\data\platerecognition\chars2';
datadir='E:\PatternRecognition\PlateRecognition\EasyPR\EasyPR-1.4\resources\train\ann';
opts.expDir = fullfile(vl_rootnn, 'data', 'plate-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
if exist(opts.imdbPath,'file')
    imdb=load(opts.imdbPath);
else
    imdb=cnn_plate_setup_data(datadir);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end
net=cnn_plate_init();
net.meta.normalization.averageImage =imdb.images.data_mean ;
opts.train.gpus=1;
[net, info] = cnn_train(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;


function fn = getBatch(opts)
% --------------------------------------------------------------------
    fn = @(x,y) getSimpleNNBatch(x,y) ;
end
function [images, labels]  = getSimpleNNBatch(imdb, batch)
    images = imdb.images.data(:,:,:,batch) ;
    labels = imdb.images.labels(1,batch) ;
    if opts.train.gpus > 0
        images = gpuArray(images) ;
    end
end
end
