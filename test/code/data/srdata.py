import os
import random

from data import common
from data import utils_image

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms
from scipy.ndimage import convolve



class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self._set_filesystem(args.dir_data)
        #self._get_blur_kernel(args)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]


        # self.images_hr 和 self.images_lr  全是文件名列表
        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:  #sep 将.png替换为.npy
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)

                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        ##这里还没处理
        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _get_blur_kernel(self,args):
        #motion_path = args.motion_path
        #motion_kernels = np.load("motion_blur_kernel.npy")
        #blur_kernel_list = [motion_kernels]
        blur_kernel_list = []
        scale = self.scale[self.idx_scale]
        top = scale * 10 + 1
        for i in range(2, top, 1):
            for j in range(2, top, 1):
                kernel = np.reshape(common.anisogkern(15, i/10, j/10),(1,15,15))
                blur_kernel_list.append(kernel)
        self.blur_kernels = np.reshape(np.concatenate(blur_kernel_list,axis=0),(-1,15,15,1))
        print("load blur kernels {}".format(self.blur_kernels.shape))


    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    # dataset 必须实现的方法 #
    def __getitem__(self, idx): #idx 第几张图片的意思
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr, hr, degrade = self._sr_down(lr,hr)

        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename, degrade

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    # 得到一张图片中的一小片  48  96
    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size  # 96
        scale = self.scale[self.idx_scale]  # 2
        multi_scale = len(self.scale) > 1  # 1只有一个尺度的时候
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            # lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = hr.shape[0:2] # 低分辨率的高 和 宽
            ih = ih // scale
            iw = iw // scale
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

    def _sr_down(self,lr, hr):

        if self.train:
            scale = self.scale[self.idx_scale]
            top  = scale * 10 + 1
            std1 = random.randrange(0,top,1)
            std2 = random.randrange(0,top,1)
            if std1 <=1 or std2 <=1:
                kernel = np.zeros((15, 15, 1), np.float32)
                kernel[7, 7, 0] = 1
                noise_std = 0
            else:
                kernel = common.anisogkern(15, std1/10, std2/10).reshape(15,15,1)
                noise_std = random.randrange(0, 71, 1)
                blur_hr = convolve(hr, kernel, mode='nearest')
                lr = utils_image.imresize_np(blur_hr,1/scale).astype(np.uint8)
        else:
            kernel = np.zeros((15, 15, 1), np.float32)
            kernel[7, 7, 0] = 1
            noise_std = 0

        '''
        if self.train:
            if random.random()>=0.3:
                blur_type = 1
                index = random.randrange(0,self.blur_kernels.shape[0],1)
                kernel = self.blur_kernels[index,:,:,:]
                noise_std = random.randrange(0, 71, 1)
            else:
                blur_type = 0
                kernel = np.zeros((15, 15, 1), np.float32)
                kernel[7, 7, 0] = 1
                noise_std = 0
        else:
            blur_type = 0
            kernel = np.zeros((15, 15, 1), np.float32)
            kernel[7,7,0] = 1
            #kernel = self.blur_kernels[80,:,:,:]
            noise_std = 0

        if blur_type == 1:

            scale = self.scale[self.idx_scale]
            ih, iw = hr.shape[0:2]
            ih = ih // scale
            iw = iw // scale
            blur_down = transforms.Compose([
                    transforms.Lambda(lambda x: convolve(x, kernel, mode='nearest')),
                    transforms.ToPILImage(),
                    transforms.Resize((ih, iw)),
                ])

            lr = np.array(blur_down(hr))
        '''

        noise = np.random.normal(scale=noise_std, size=lr.shape).round()
        lr = lr.astype(np.int16) + noise.astype(np.int16)
        lr = lr.clip(0, 255).astype(np.uint8)

        kernel = np.reshape(kernel, 225).astype(np.float32)
        kernel = kernel * 255 - 127.5

        noise_std = np.array([noise_std]).astype(np.float32)
        degrade = np.concatenate((kernel,noise_std),axis=0) #226
        return lr,hr,degrade


