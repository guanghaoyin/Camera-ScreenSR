import os

from Data import utils
from Data import srdata

import numpy as np
import imageio

import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.dir_hr = args.dir_hr
        self.dir_lr = args.dir_lr
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.scale = args.scale

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
            # align test
            # idx_begin = 800
            # idx_end = self.args.n_train + 100
        else:
            idx_begin = self.args.n_train
            # idx_end = self.args.offset_val + self.args.n_val
            idx_end = self.args.n_train + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            # filename_lr = '{:0>4}'.format(i) + '_align'
            filename_lr = '{:0>4}'.format(i)
            filename_hr = '{:0>4}'.format(i)
            # list_hr.append(os.path.join(self.dir_hr, filename_hr + '.JPG'))
            list_hr.append(os.path.join(self.dir_hr, filename_hr + '.png'))
            # list_lr.append(os.path.join(self.dir_lr, filename_lr + self.ext))

            for si, s in enumerate(self.scale):
                # list_lr[si].append(os.path.join(self.dir_lr,filename_lr + 'x' + str(self.scale) + '.png'))
                list_lr[si].append(os.path.join(self.dir_lr,filename_lr + '.png'))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data, dir_hr,dir_lr):
        if self.train:
            self.apath = dir_data + '/DIV2K_train'
        else:
            self.apath = dir_data + '/DIV2K_test'
        # self.apath = dir_data + '/DIV2K_test'
        self.dir_hr = os.path.join(self.apath, dir_hr)
        self.dir_lr = os.path.join(self.apath, dir_lr + self.scale)

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx