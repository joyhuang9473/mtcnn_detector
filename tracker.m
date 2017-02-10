clear;

result_path = '/workspace/datasets/LFW/intermediate';
flmk = fopen(fullfile(result_path, 'mtcnn_v2_landmark_list.txt'), 'w');
floss = fopen(fullfile(result_path, 'mtcnn_v2_loss_list.txt'), 'w');
ferror = fopen(fullfile(result_path, 'mtcnn_v2_error_list.txt'), 'w');

%list of images
imglist=importdata(fullfile(result_path, 'raw_image_list.txt'));

%path of toolbox
caffe_path='/opt/caffe/matlab';
pdollar_toolbox_path='./dependency/toolbox';
caffe_model_path='./model';
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7];

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);

for i=1:length(imglist)
    if ~exist(imglist{i}, 'file')
        disp(['file not exist:' imglist{i}])
        fprintf(ferror, '%s\n', imglist{i});
        continue;
    end
    
    try
	    origin_img=imread(imglist{i});

        if size(origin_img,3)==3
            img=origin_img;
        elseif size(origin_img, 3)==1
            img=origin_img(:,:,[1 1 1]);
        else
            throw(MException('color channels are not valid'));
        end
    catch exception
        disp(['file not valid:' imglist{i}])
        fprintf(ferror, '%s\n', imglist{i});
        continue;
    end
	%we recommend you to set minsize as x * short side
	minl=min([size(img,1) size(img,2)]);
	minsize=fix(minl*0.1);

    [boudingboxes, points]=detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);

    %faces{i,1}={boudingboxes};
	%faces{i,2}={points'};
	%show detection result
	numbox=size(boudingboxes,1);

    if isempty(points)
        disp(['face not detected:' imglist{i}])
        fprintf(floss, '%s\n', imglist{i});
        continue;
    end
    
    %get face 1
    numbox=1;
    
	for j=1:numbox
        disp(['success:' imglist{i}])
        %write file path
        fprintf(flmk, '%s', imglist{i});
        %write 5 landmarks
        for k = 1:5
            fprintf(flmk, ' %.3f %.3f', points(k,1), points(k+5,1));
        end
        
        fprintf(flmk, '\n');
    end
    
end

fclose(flmk);
fclose(floss);
fclose(ferror);
clear;
