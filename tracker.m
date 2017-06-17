clear;

setenv('LC_ALL', 'en_US.UTF-8');

[FILE_LIST, DATA_ROOT, INFO_ROOT, NORMAL_LIST, MULTI_LIST, LOSS_LIST, ERROR_LIST] = config();

% log for bbox
fnormal_bbox = fopen([NORMAL_LIST '.bbox'], 'w');
fmulti_bbox = fopen([MULTI_LIST '.bbox'], 'w');
floss_bbox = fopen([LOSS_LIST '.bbox'], 'w');
ferror_bbox = fopen([ERROR_LIST '.bbox'], 'w');

% log for lmk
fnormal_lmk = fopen([NORMAL_LIST '.lmk'], 'w');
fmulti_lmk = fopen([MULTI_LIST '.lmk'], 'w');
floss_lmk = fopen([LOSS_LIST '.lmk'], 'w');
ferror_lmk = fopen([ERROR_LIST '.lmk'], 'w');

%list of images
imglist = importdata(FILE_LIST);

%path of toolbox
caffe_path = '/opt/caffe/matlab';
pdollar_toolbox_path = './dependency/toolbox';
matlab_json_path = './dependency/matlab-json';
caffe_model_path = './model';

addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));
addpath(genpath(matlab_json_path));

gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

% load json plugin
% ref: https://github.com/kyamagu/matlab-json
json.startup

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

% create info structure
info = {};

for i = 1:length(imglist)
    disp(['process:' num2str(i) '/' num2str(length(imglist))]);

    imgPath = imglist{i};
    absImgPath = fullfile(DATA_ROOT, imgPath);
    [identity_name, file_name_part, ext] = fileparts(imgPath);

    % add file_name and identity_name to info
    info.file_name = [file_name_part ext];
    info.identity_name = identity_name;

    if ~exist(absImgPath, 'file')
        disp(['file not exist:' imgPath])
        fprintf(ferror_bbox, '%s\n', imgPath);
        fprintf(ferror_lmk, '%s\n', imgPath);
        continue;
    end

    % prevent JPEG library error (8 bit) crash
    try
        I = rgb2gray(imread(absImgPath));
        [w, l] = size(I);
        gray_percent = sum(sum(I==128))/(w*l);;

        if gray_percent > 0.07
            disp(['file not valid:' imgPath])
            fprintf(ferror_bbox, '%s\n', imgPath);
            fprintf(ferror_lmk, '%s\n', imgPath);
            continue
        end
    catch
        disp(['file not valid:' imgPath])
        fprintf(ferror_bbox, '%s\n', imgPath);
        fprintf(ferror_lmk, '%s\n', imgPath);
        continue
    end

    try
        origin_img = imread(absImgPath);
        [height, width, channel] = size(origin_img);

        info.width = width;
        info.height = height;
        info.channel = channel;

        if channel == 3
            img = origin_img;
        elseif channel == 1
            img = origin_img(:,:,[1 1 1]);
        else
            throw(MException('color channels are not valid'));
        end
    catch exception
        disp(['file not valid:' imgPath])
        fprintf(ferror_bbox, '%s\n', imgPath);
        fprintf(ferror_lmk, '%s\n', imgPath);
        continue;
    end

    %we recommend you to set minsize as x * short side
    minl = min([size(img,1) size(img,2)]);
    minsize = fix(minl*0.1);

    [bboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, false, factor);

    %% add bbox to info
    numbox = size(bboxes, 1);
    info.boundingbox.selected_index = '';
    info.boundingbox.faces = {};
    info.boundingbox.nums = numbox;

    if numbox == 0
        disp(['bbox not detected:' imgPath])
        fprintf(floss_bbox, '%s\n', imgPath);
    elseif numbox == 1
        info.boundingbox.selected_index = 0;
        fprintf(fnormal_bbox, '%s\n', imgPath);
    else
        disp(['multi-bboxes detected:' imgPath])
        fprintf(fmulti_bbox, '%s\n', imgPath);
    end

    for j = 1:numbox
        x = bboxes(j, 1);
        y = bboxes(j, 2);
        width = bboxes(j, 3) - bboxes(j, 1);
        height = bboxes(j, 4) - bboxes(j, 2);

        face_index = ['face_' num2str(j-1)];
        info.boundingbox.faces.(face_index) = struct( ...
            'x', x, ...
            'y', y, ...
            'width', width, ...
            'height', height);
    end

    %% add landmark to info
    numlmk = size(points, 2);
    info.landmark.selected_index = '';
    info.landmark.faces = {};
    info.landmark.nums = numlmk;

    if numlmk == 0
        disp(['lmk not detected:' imgPath])
        fprintf(floss_lmk, '%s\n', imgPath);
    elseif numlmk == 1
        info.landmark.selected_index = 0;
        fprintf(fnormal_lmk, '%s\n', imgPath);
    else
        disp(['multi-lmks detected:' imgPath])
        fprintf(fmulti_lmk, '%s\n', imgPath);
    end

    for j = 1:numlmk
        lmk_points = {};

        for k = 1:5
            lmk_points = [lmk_points, points(k, j), points(k+5, j)];
        end

        face_index = ['face_' num2str(j-1)];
        info.landmark.faces.(face_index) = struct( ...
            'points', {lmk_points}, ...
            'left_eye_index', 0, ...
            'right_eye_index', 1, ...
            'nose_eye_index', 2, ...
            'left_mouse_index', 3, ...
            'right_mouse_index', 4);
    end

    % write info to json
    id_info_root = fullfile(INFO_ROOT, identity_name);
    if ~exist(id_info_root, 'dir')
        mkdir(id_info_root);
    end

    % add pretain column (valid, mean_distance) to info
    info.valid = true;
    info.mean_distance = '';

    json.write(info, fullfile(id_info_root, [file_name_part ext '.json']));

end

fclose(fnormal_bbox);
fclose(fmulti_bbox);
fclose(floss_bbox);
fclose(ferror_bbox);
fclose(fnormal_lmk);
fclose(fmulti_lmk);
fclose(floss_lmk);
fclose(ferror_lmk);
clear;
