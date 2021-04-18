import os
from importlib import import_module

import matplotlib

import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self,args,ckp):
        super(Loss,self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            #1*L1 + ...
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                #1*VGG54
                module = import_module('Loss.vgg_loss')
                loss_function = getattr(module, 'VGGLoss')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('Color') >= 0:
                module = import_module('Loss.color_loss')
                loss_function = getattr(module, 'ColorLoss')()
            elif loss_type.find('TV') >= 0:
                module = import_module('Loss.tv_loss')
                loss_function = getattr(module, 'TVLoss')()

            else:
                raise TypeError("There is no define about loss function")

            #loss append
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        #loss module append
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        #set device
        device = torch.device('cpu' if args.cpu else 'cuda:'+str(args.GPU_ID))
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half() ##nn.MOdule.loss.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr, other = None):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def plot_loss_fig(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))