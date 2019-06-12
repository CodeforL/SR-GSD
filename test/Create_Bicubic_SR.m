function Evaluate_PSNR_SSIM()

clear all; close all; clc

% set path
% method /
degradation = 'BI';
methods = {'BICUBIC'};
dataset = {'Set5','Set14','B100','Manga109','Urban100'};
blur = {'BI_0','BI_1','BI_2'}
ext = {'*.jpg', '*.png', '*.bmp'};
num_method = length(methods);
num_set = length(dataset);
num_blur = length(blur)

if strcmp(degradation, 'BI') 
    scale_all = [2];
else
    scale_all = 3;
end

for idx_method = 1:num_method
    for idx_set = 1:num_set
        fprintf('********************************************\n');
        fprintf('Method_%d: %s; Set: %s\n', idx_method, methods{idx_method}, dataset{idx_set});
        for scale = scale_all
            for id_blur = 1:num_blur
                filepaths = [];
                for idx_ext = 1:length(ext)
                    filepaths = cat(1, filepaths, dir(fullfile('./LR', dataset{idx_set}, ['x', num2str(scale)], blur{id_blur}, ext{idx_ext})));
                end
                folder_SR = fullfile('./SR',methods{idx_method}, dataset{idx_set}, ['x', num2str(scale)],blur{id_blur});
                if ~exist(folder_SR)
                    mkdir(folder_SR)
                end
                fprintf('********************************************\n');
                fprintf('x%d bi_blur:%s\n', scale, blur{id_blur});
                for idx_im = 1:length(filepaths)
                    name_LR = filepaths(idx_im).name;
                    name_SR = strrep(name_LR, 'LRBI', methods{idx_method});
                    fprintf('%s\n', name_SR);
                    im_LR = imread(fullfile('./LR', dataset{idx_set}, ['x', num2str(scale)], blur{id_blur}, name_LR));
                    im_SR = imresize(im_LR, scale, 'bicubic');
                    fn_SR = fullfile('./SR', methods{idx_method}, dataset{idx_set}, ['x', num2str(scale)], blur{id_blur} ,name_SR);
                    imwrite(im_SR, fn_SR, 'png');
                end
            end
        end
    end
end
end
