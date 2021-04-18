import os

from Data import utils

import numpy as np
import imageio

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data, args.dir_hr, args.dir_lr)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = imageio.imread(v)
                    name_sep = v.replace(self.args.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = imageio.imread(v)
                        name_sep = v.replace(self.args.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.args.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.args.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data, dir_hr, dir_lr):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr, filename)
        lr, hr = utils.set_channel([lr, hr], self.args.n_colors)
        # transforms_ = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        # lr = transforms_(lr)
        # hr = transforms_(hr)
        lr_tensor, hr_tensor = utils.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        from  PIL import Image
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        # if self.args.mix_lr:
        #     if self.train:
        #         import random
        #         if random.random() < 0.2:
        #             img_name = lr[len(self.apath)+len(self.args.dir_lr)+3:]
        #             lr = os.path.join(self.apath, self.args.dir_lr + self.scale + '_Gen', img_name)
        #     else:
        #         pass
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = imageio.imread(lr)
            if lr.shape[0] < self.args.patch_size or lr.shape[1] < self.args.patch_size:
                print(filename)
            hr = imageio.imread(hr)
            lr = np.array(lr)
            hr = np.array(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        lr = utils.normalize(lr)
        hr = utils.normalize(hr)
        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr, filename):
        patch_size = self.args.patch_size
        scale = int(self.scale[self.idx_scale])
        name = filename
        # multi_scale = len(self.scale) > 1
        multi_scale = int(self.scale) > 1
        if self.train:
            lr, hr = utils.get_patch(
                name, lr, hr, patch_size, int(scale), multi_scale=multi_scale
            )
            lr, hr = utils.augment([lr, hr])
            # lr = utils.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
