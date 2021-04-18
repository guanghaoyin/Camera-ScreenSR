#Only for evaluatipn
import os
from Data import srdata

class BSD100(srdata.SRData):
    def __init__(self, args, train=False):
        super(BSD100, self).__init__(args, train)
        self.dir_hr = args.dir_hr
        self.dir_lr = args.dir_lr
        self.scale = args.scale

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            raise NotImplementedError("BSD100 dataset can not be used for training!")
        else:
            filename_hr = []
            filename_lr = []
            for i in range(100):
                filename_hr.append('BSD_img_' + '%03d'%(i+1))
                filename_lr.append('BSD_img_' + '%03d'%(i+1))
                list_hr.append(os.path.join(self.dir_hr,filename_hr[i] + '.png'))
                for si, s in enumerate(self.scale):
                    list_lr[si].append(os.path.join(self.dir_lr,filename_lr[i] + '.png'))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data, dir_hr, dir_lr):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, dir_hr)
        self.dir_lr = os.path.join(self.apath, dir_lr + self.scale)
        self.ext = '.jpg'

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx