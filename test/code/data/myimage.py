import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0

        apath = args.testpath + '/' + args.testset + '/x' + str(args.scale[0]) +'/' + args.bi_blur_path
        print('test '+ apath)
        #bi_blur = ['BI_0','BI_0.2','BI_1','BI_2']
        self.filelist = []
        self.imnamelist = []
        if not train:
            for f in os.listdir(apath):
                try:
                    filename = os.path.join(apath, f)
                    misc.imread(filename)
                    self.filelist.append(filename)
                    self.imnamelist.append(f)
                except:
                    pass

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist[idx])
        lr = common.set_channel(lr, self.args.n_colors)
        return common.np2Tensor([lr], self.args.rgb_range)[0], -1, filename, -1


    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

