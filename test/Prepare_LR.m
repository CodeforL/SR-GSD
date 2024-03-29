function Prepare_TestData_HR_LR()
clear all; close all; clc
path_original = './OriginalTestData';
% dataset  = {'Set5','Set14','B100','Manga109','Urban100'};
dataset  = {'Set5'};
ext = {'*.jpg', '*.png', '*.bmp'};

degradation = 'BI'; % BI, BD, DN
if strcmp(degradation, 'BI') 
    scale_all = [2,3,4,8];
else
    scale_all = 3;
end
degrade_std = [0,2,50];

for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        im_ori = imread(fullfile(path_original, dataset{idx_set}, name_im));
        if size(im_ori, 3) == 1
            im_ori = cat(3, im_ori, im_ori, im_ori);
        end
        for scale = scale_all
            for std = degrade_std
                im_HR = modcrop(im_ori, scale);
                im_LR = imresize_BD(im_HR,scale,'Gaussian',std);
                folder_HR = fullfile('./HR', dataset{idx_set}, ['x', num2str(scale)]);
                folder_LR = fullfile('./LR', dataset{idx_set}, ['x', num2str(scale)],['BI_',num2str(std)]);
                if ~exist(folder_HR)
                    mkdir(folder_HR)
                end
                if ~exist(folder_LR)
                    mkdir(folder_LR)
                end
                % fn
                fn_HR = fullfile('./HR', dataset{idx_set}, ['x', num2str(scale)], [name_im(1:end-4), '_HR_x', num2str(scale), '.png']);
                fn_LR = fullfile('./LR', dataset{idx_set}, ['x', num2str(scale)], ['BI_',num2str(std)],[name_im(1:end-4), '_LR', degradation, '_x', num2str(scale), '.png']);
                imwrite(im_HR, fn_HR, 'png');
                imwrite(im_LR, fn_LR, 'png');
            end
        end
        fprintf('\n');
    end
    fprintf('\n');
end
end
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end


function [LR] = imresize_BD(im, scale, type, sigma)
    if sigma == 0
        LR = imresize(im, 1/scale, 'bicubic');
    end
    if sigma == 2
        kernelsize = 15;
        kernel  = fspecial('gaussian',kernelsize,sigma);
        blur_HR = imfilter(im,kernel,'replicate');
        LR = imresize(blur_HR, 1/scale, 'bicubic');
    end
    if sigma == 50
        kernelsize = 15;
        kernel  = fspecial('gaussian',kernelsize,2);
        blur_HR = imfilter(im,kernel,'replicate');
        LR = imresize(blur_HR, 1/scale, 'bicubic');
        LR = single(LR);
        LR = LR + single(sigma*randn(size(LR)));
        LR = uint8(LR);
    end
end

function ImLR = imresize_DN(ImHR, scale, sigma)
% ImLR and ImHR are uint8 data
% downsample by Bicubic
ImDown = imresize(ImHR, 1/scale, 'bicubic'); % 0-255
ImDown = single(ImDown); % 0-255
ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
ImLR = uint8(ImDownNoise); % 0-255
end
