import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        kernel = self.gauss_kernel(21, 3, 3)
        kernel = torch.from_numpy(kernel).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)#0.333333
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        out_filter = np.repeat(out_filter, channels, axis=3)
        return out_filter

    def forward(self, sr, hr):
        sr = F.conv2d(sr, self.weight, stride=1, padding=10)
        hr = F.conv2d(hr, self.weight, stride=1, padding=10)
        return torch.sum(torch.pow((sr - hr), 2)).div(2 * sr.size()[0])
if __name__ == '__main__':
    cl = ColorLoss()