from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.transforms as transforms

import skimage.color as sc
import random
import datetime
import matplotlib.pyplot as plt
import imageio
import time

# path = '../../../../Dataset/SISR/DIV2K/DIV2K_test'
path = 'D:/dataset/SISR'
path = path + '/DIV2K/DIV2K_train'
# path = path + '/Urban100'
#timer
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

#align
def get_fila_dataname(path, file):
    list_path = os.path.join(path, file)
    list_dir = os.listdir(list_path)
    total = len(list_dir)
    dataname = []
    for i in tqdm(range(total), ascii=True, desc= 'load data from '+ list_path):
        dir = list_dir[i]
        # datapath = os.path.join(list_path, dir)
        dataname.append(dir)
    return dataname

def change_dataname(path, file, dataname):
    list_path = os.path.join(path, file)
    list_dir = os.listdir(list_path)
    total = len(list_dir)
    for i in tqdm(range(total), ascii=True, desc='change data from ' + list_path):
        dir = list_dir[i]
        old = os.path.join(list_path, dir)
        new = os.path.join(list_path, dataname[i][:-3] + dir[-3:])
        os.rename(old,new)

def align_name(path, file1, file2):
    dataname = get_fila_dataname(path,file1)
    print(dataname)
    change_dataname(path, file2, dataname)


#image
def normalize(image):
    return (image / 127.5) - 1
    # return image / 255

def unnormalize(image):
    return (image + 1) * 127.5
    # return image * 255

def rgb2ycbcr(rgb_img):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    offset = np.array([16, 128, 128])
    ycbcr_img = np.zeros(rgb_img.shape)
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

def ycbcr2rgb(ycbcr_img):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.array([16, 128, 128])
    rgb_img = np.zeros(ycbcr_img.shape, dtype=np.uint8)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            [r, g, b] = ycbcr_img[x,y,:]
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255, np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))

def get_patch(name, img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]

    p = scale if multi_scale else 1
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    # print(name)
    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    return img_in, img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

        curPath = os.path.abspath(os.path.dirname(__file__))
        rootPath = curPath[:curPath.find("Data")]
        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = os.path.join(rootPath, 'experiment', args.save)
            # self.dir = './experiment/' + args.save
        else:
            self.dir = os.path.join(rootPath, 'experiment', args.load)
            # self.dir = './experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, model, loss, optimizer, epoch, is_best=False):
        model.save(self.dir, epoch, is_best=is_best)
        loss.save(self.dir)
        # loss.plot_loss_fig(self.dir, epoch)

        # self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )
        if not os.path.exists(self.dir + '/model.txt'):
            open_type = 'w'
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
            with open(self.dir + '/model.txt', open_type) as f:
                f.write(now + '\n\n')
                f.write(str(model))
                f.write('\n')

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, epoch, filename, save_list, scale, postfix = ('SR', 'LR', 'HR')):
        # filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        filename = '{}/results/{}'.format(self.dir, filename)
        for v, p in zip(save_list, postfix):
            if epoch == 0 or p == 'SR' or p == 'DUR_SR':
                normalized = v[0].data.mul(255 / self.args.rgb_range)
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                imageio.imsave('{}{}{}.png'.format(filename, p,epoch), ndarr)
                # imageio.imsave('{}.png'.format(filename), ndarr)
            else:
                continue

    def save_results(self, epoch, filename, save_list, scale, postfix=('SR', 'LR', 'HR'), set_fix=True):
        for v, p in zip(save_list, postfix):
            if epoch == 0 or p == 'SR' or p == 'DUR_SR':
                normalized = v[0].data.mul(255 / self.args.rgb_range)
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                if set_fix:
                    filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
                    imageio.imsave('{}{}{}.png'.format(filename, p, epoch), ndarr)
                else:

                    imageio.imsave('{}.png'.format(filename), ndarr)
            else:
                continue


#model
def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    kwargs = {}
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    else:
        raise NotImplementedError("There is no define of optimizer")

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

#Tensor
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

def tensor_prepare(l,args, volatile=False):
    device = torch.device('cpu' if args.cpu else 'cuda:'+str(args.GPU_ID))

    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

#load model
def load_GPU(model,model_path):
    from option import args
    device = torch.device('cuda:' + str(args.GPU_ID))
    state_dict = torch.load(model_path, map_location=device)
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(state_dict)
    return model

def load_GPUS(model,model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model
# align_name(path, 'DIV2K_valid_HR', 'DIV2K_dark_canon_LR')
# align_name(path, 'Urban100_HR', 'Urban100_align_canon_LR')
# align_name(path, 'DIV2K_valid_HR', 'DIV2K_align_canon_LR')
# align_name(path, 'DIV2K_valid_HR', 'DIV2K_valid_iphone7_MOV')