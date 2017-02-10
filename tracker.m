clear;

[FILE_LIST, DATA_ROOT, LMK_LIST, MULTI_LMK_LIST, LOSS_LIST, ERROR_LIST] = config();

flmk = fopen(LMK_LIST, 'w');
fmulti = fopen(MULTI_LMK_LIST, 'w');
floss = fopen(LOSS_LIST, 'w');
ferror = fopen(ERROR_LIST, 'w');

%list of images
imglist = importdata(FILE_LIST);

%path of toolbox
caffe_path = '/opt/caffe/matlab';
pdollar_toolbox_path = './dependency/toolbox';
caffe_model_path = './model';

addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

%three steps's threshold
threshold = [0.6 0.7 0.7];

%scale factor
factor = 0.709;

%load caffe models
prototxt_dir = strcat(caffe_model_path, '/det1.prototxt');
model_dir = strcat(caffe_model_path, '/det1.caffemodel');
PNet = caffe.Net(prototxt_dir, model_dir, 'test');
prototxt_dir = strcat(caffe_model_path, '/det2.prototxt');
model_dir = strcat(caffe_model_path, '/det2.caffemodel');
RNet = caffe.Net(prototxt_dir, model_dir, 'test');
prototxt_dir = strcat(caffe_model_path, '/det3.prototxt');
model_dir = strcat(caffe_model_path, '/det3.caffemodel');
ONet = caffe.Net(prototxt_dir, model_dir, 'test');
prototxt_dir = strcat(caffe_model_path, '/det4.prototxt');
model_dir = strcat(caffe_model_path, '/det4.caffemodel');
LNet = caffe.Net(prototxt_dir, model_dir, 'test');

for i = 1:length(imglist)
    imgPath = fullfile(DATA_ROOT, imglist{i});

    if ~exist(imgPath, 'file')
        disp(['file not exist:' imgPath])
        fprintf(ferror, '%s\n', imgPath);
        continue;
    end

    try
        origin_img = imread(imgPath);

        if size(origin_img, 3) == 3
            img = origin_img;
        elseif size(origin_img, 3) == 1
            img = origin_img(:,:,[1 1 1]);
        else
            throw(MException('color channels are not valid'));
        end
    catch exception
        disp(['file not valid:' imgPath])
        fprintf(ferror, '%s\n', imgPath);
        continue;
    end
    %we recommend you to set minsize as x * short side
    minl = min([size(img,1) size(img,2)]);
    minsize = fix(minl*0.1);

    [bboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, false, factor);

    %show detection result
    numbox = size(bboxes, 1);

    if isempty(points)
        disp(['lmk not detected:' imgPath])
        fprintf(floss, '%s\n', imgPath);
        continue;
    end

    if numbox > 1
        for j = 1:numbox
            disp(['multi face:' imgPath])
            fprintf(fmulti, '%s', imgPath);

            for k = 1:5
                fprintf(fmulti, '\t%.3f\t%.3f', points(k,j), points(k+5,j)); % (x1, y1)
            end

            %%TODO: bbox
            fprintf(fmulti, '\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f', bboxes(j,1), bboxes(j,2), bboxes(j,3), bboxes(j,4), bboxes(j,5));

            fprintf(fmulti, '\n');
        end
    elseif numbox == 1
        disp(['success:' imgPath])
        fprintf(flmk, '%s', imgPath);

        for k = 1:5
            fprintf(flmk, '\t%.3f\t%.3f', points(k,1), points(k+5,1)); % (x1, y1)
        end

        %%TODO: bbox
        fprintf(flmk, '\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f', bboxes(1,1), bboxes(1,2), bboxes(1,3), bboxes(1,4), bboxes(1,5));

        fprintf(flmk, '\n');
    else
        disp(['something wrong:' imgPath])
        fprintf(ferror, '%s\n', imgPath);
    end

end

fclose(flmk);
fclose(fmulti);
fclose(floss);
fclose(ferror);
clear;
