import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.chop = args.chop
        self.save_results = args.save_results
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda:'+str(args.GPU_ID))
        self.n_GPUs = args.n_GPUs
        # self.save_models = args.save_models

        module = import_module('Model.' + args.model)
        self.model = module.make_model(args).to(self.device)

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(self.n_GPUs))
        if args.precision == 'half': self.model.half()
        if args.print_model: print(self.model)

    def forward(self, x, idx_scale = 2):
        if self.n_GPUs == 1:
            model = self.model
        else:
            model = self.model.module
        if hasattr(model, 'set_scale'):
            model.set_scale(idx_scale)
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        # if self.save_models:
        if self.save_results:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )
    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )