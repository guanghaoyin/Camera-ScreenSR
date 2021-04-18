import torch
import torch.nn as nn

class DuRB_p(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, res_dim=32, k1_size=3, k2_size=1, dilation=1, norm_type="batch_norm",
                 with_relu=True):
        super(DuRB_p, self).__init__()

        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)

        # T^{l}_{1}: (conv.)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)

        # T^{l}_{2}: (conv.)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=1)

        self.with_relu = with_relu
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x

        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += x_r
        x = self.relu(x)

        # T^{l}_{1}
        x = self.up_conv(x)
        x += res
        x = self.relu(x)
        res = x

        # T^{l}_{2}
        x = self.down_conv(x)
        x += x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res


# ---------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect.")

    def forward(self, x):
        out = self.norm(x)
        return out


class InsNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InsNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        flat_len = x.size(2) * x.size(3)
        vec = x.view(x.size(0), x.size(1), flat_len)
        mean = torch.mean(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((flat_len - 1) / float(flat_len))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out