import torch
import torch.nn as nn
import torch.nn.functional as F

class LapLoss(nn.Module):
    def __init__(self):
        super(LapLoss,self).__init__()
        kernel = self.lap_kernel(3,3,3)

    def forward(self, sr,hr):
        batch_size, c, h, w = sr.size()
        chw = c*h*w
        count_h = self._tensor_size(sr[:, :, 2:, :])
        count_w = self._tensor_size(sr[:, :, :, 2:])
        h_lap = sr[:,:,2:,:] + sr[:,:,:-2,:] - 2 * sr[:,:,1:-1,:]
        w_lap = sr[:,:,:,2:] + sr[:,:,:,:-2] - 2 * sr[:,:,:,1:-1]
        return 1/chw * (h_lap/count_h + w_lap/count_w)/batch_size

    def _tensor_size(self,t):
            return t.size()[1] * t.size()[2] * t.size()[3]
