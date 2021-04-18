import torch.nn as nn
import torch
from Model.Attention_module import NONLocal2D, CALayer2D
from Model.DURB import DuRB_p
import math
import os
import Data.utils as utils

class DURAG(nn.Module):
    def __init__(self, in_dim=64, out_dim=64, res_dim=64, k1_size=3, k2_size=1, dilation=1, norm_type="batch_norm",
                 with_relu=True):
        super(DURAG,self).__init__()
        self.DuRB_p = DuRB_p(in_dim, out_dim, res_dim, k1_size, k2_size, dilation, norm_type, with_relu)
        # self.CA = CALayer2D(channel=out_dim)
        # self.Non_local = NONLocal2D(in_feat=out_dim, inter_feat=out_dim // 2)

    def forward(self, x, res):
        x, res = self.DuRB_p(x, res)
        # x = self.Non_local(x)
        # x = self.CA(x)
        return x, res

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        # x = self.prelu(x)
        return x

class DURCAN(nn.Module):
    def __init__(self,args):
        super(DURCAN,self).__init__()
        self.input_channel = 3
        self.output_channel = 3
        self.inter_channel = 64
        self.n_GPUs = args.n_GPUs
        self.up_scale = int(args.scale)
        self.upsample_block_num = int(math.log(self.up_scale, 2))
        self.save_results = args.save_results

        # self.origin_conv = nn.Sequential(nn.Conv2d(self.input_channel, self.inter_channel, kernel_size=3, stride=1, padding=1, bias=True),nn.ReLU())
        self.origin_conv = nn.Sequential(nn.Conv2d(self.input_channel, self.inter_channel, kernel_size=3, stride=1, padding=1, bias=True))
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = utils.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = utils.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.Attention_begin = CALayer2D(channel=self.inter_channel)
        # self.DURAG1 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=3, k2_size=3, dilation=1)
        self.DURAG1 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=5, k2_size=3, dilation=1)
        # self.DURAG2 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=3, k2_size=3, dilation=1)
        self.DURAG2 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=5, k2_size=3, dilation=1)
        # self.DURAG3 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=5, k2_size=3, dilation=1)
        self.DURAG3 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=5, dilation=1)
        # self.DURAG4 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=5, k2_size=3, dilation=1)
        self.DURAG4 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=5, dilation=1)
        # self.DURAG5 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=5, dilation=1)
        self.DURAG5 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=11, k2_size=7, dilation=1)

        # self.DURAG6 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=3, dilation=1)
        self.DURAG6 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=11, k2_size=7, dilation=1)

        # self.DURAG7 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=3, dilation=1)
        self.DURAG7 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=11, k2_size=7, dilation=1)

        # self.DURAG8 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=5, dilation=1)
        self.DURAG8 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=11, k2_size=7, dilation=1)
        # self.DURAG9 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=5, k2_size=3, dilation=1)
        self.DURAG9 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=11, k2_size=5, dilation=1)
        # self.DURAG10 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=5, k2_size=3, dilation=1)
        self.DURAG10 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=11, k2_size=5, dilation=1)
        # self.DURAG11 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=3, k2_size=3, dilation=1)
        self.DURAG11 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=5, dilation=1)
        # self.DURAG12 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=3, k2_size=3, dilation=1)
        self.DURAG12 = DURAG(in_dim=self.inter_channel, out_dim=self.inter_channel, res_dim=self.inter_channel, k1_size=7, k2_size=5, dilation=1)
        self.Attention_end = CALayer2D(channel=self.inter_channel)
        #original
        self.last_conv =  nn.Sequential(nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=3, stride=1, padding=1, bias=True))
        self.upsample1 = [UpsampleBLock(self.inter_channel, 2) for _ in range(self.upsample_block_num)]
        self.upsample1.append(nn.Conv2d(self.inter_channel, self.output_channel, kernel_size=3, stride=1, padding=1, bias=True))
        self.upsample1.append(nn.Tanh())
        self.upsample1 = nn.Sequential(*self.upsample1)


        # self.mid_conv =  nn.Sequential(nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=3, stride=1, padding=1, bias=True))

        # self.last_conv = nn.Conv2d(self.inter_channel, self.output_channel, kernel_size=9, padding=4)
        # self.upsample = [UpsampleBLock(self.output_channel, 2) for _ in range(self.upsample_block_num)]
        # self.upsample = nn.Sequential(*self.upsample)
        # self.final_conv = nn.Conv2d(self.output_channel, self.output_channel, kernel_size=9, padding=4)

    def forward(self, x_input):
        # x_input = self.sub_mean(x_input)
        x = self.origin_conv(x_input)
        res0 = x
        res = x

        x = self.Attention_begin(x)
        x, res = self.DURAG1(x, res)
        x, res = self.DURAG2(x, res)
        x, res = self.DURAG3(x, res)
        x, res = self.DURAG4(x, res)
        x, res = self.DURAG5(x, res)
        x, res = self.DURAG6(x, res)
        x, res = self.DURAG7(x, res)
        x, res = self.DURAG8(x, res)
        x, res = self.DURAG9(x, res)
        x, res = self.DURAG10(x, res)
        x, res = self.DURAG11(x, res)
        x, res = self.DURAG12(x, res)
        x = self.Attention_end(x)
        #original
        # x = self.last_conv(x+res0)
        x = self.last_conv(x+res)
        x = self.upsample1(x)
        # x = self.add_mean(x)

        # x = self.mid_conv(x) + res
        # x = self.last_conv(x)
        # x = self.upsample(x + x_input)
        # x = self.final_conv(x)
        return x

    # def get_model(self):
    #     if self.n_GPUs == 1:
    #         return self.model
    #     else:
    #         return self.model.module

    def save(self, apath, epoch):
        # target = self.get_model()
        torch.save(
            self.state_dict(),
            os.path.join(apath, 'model', 'DURCAN_latest.pt')
        )
        if not os.path.exists(apath + '/SR_model.txt'):
            open_type = 'w'
            with open(apath + '/model.txt', open_type) as f:
                f.write('\n\n')
                f.write(str(self))
                f.write('\n')
        if self.save_results:
            torch.save(
                self.state_dict(),
                os.path.join(apath, 'model', 'DURCAN_{}.pt'.format(epoch))
            )