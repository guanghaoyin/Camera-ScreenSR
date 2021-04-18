import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F
use_Y = True

def cal_psnr_tensor(sr:torch.Tensor, hr:torch.Tensor, scale:int, rgb_range:int, benchmark:bool = False) -> float:
    """cal_psnr

    :param sr: Tensor super resolution image from model
    :param hr: Tensor high resolution image from ground truth
    :param scale: Int scale index
    :param rgb_range: Int the range of rgb image, exp 255
    :param benchmark: Bool use benchmark test
    """
    diff = (sr - hr).data.div(rgb_range)
    # print('diff: %s'%diff)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim = 1, keepdim = True)
        #print('diff_convert: %s'%diff)
    valid = diff[:, :, shave: -shave, shave: -shave]
    #print(valid)
    mse = valid.pow(2).mean()
    #print(mse)
    return -10 * math.log10(mse)

def cal_ssim_tensor(img1, img2, data_range=255, window_size=11, window=None, full=False):
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor(np.exp(-(np.arange(window_size) - window_size // 2) ** 2 / (2.0 * sigma ** 2)))
        return gauss / gauss.sum()

    def _create_window(window_size, channel=1):
        window_1d = _gaussian(window_size, 1.5).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    """Compute Structural Similarity"""
    padd = 0  # padding for convolution
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = _create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        cs = torch.mean(v1 / v2)  # contrast sensitivity
        return ret.item(), cs.item()
    return ret.item()

def cal_psnr(hr, sr):
    # hr and sr have range [0, 255]
    hr = hr.astype(np.float64)
    sr = sr.astype(np.float64)
    mse = np.mean((hr - sr)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cal_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

if __name__ == '__main__':
    # img_hr = cv2.imread('D:/dataset/SISR/DIV2K/DIV2K_test/DIV2K_valid_HR/0802.png')
    # img_sr = cv2.imread('D:/dataset/SISR/DIV2K/DIV2K_test/DIV2K_align_canon/0802_align.jpg')
    # img_sr = cv2.imread('D:/dataset/SISR/DIV2K/DIV2K_test/DIV2K_align_iphone7/0802_align.jpg')
    # img_sr = cv2.imread('C:/Users/guanghaoyin/Documents/code/python/ImageRegistration/experiment/2019-11-19-11_06_31/results/0805_x1_SR.png')
    # img_hr = cv2.imread('C:/Users/guanghaoyin/Documents/code/python/ImageRegistration/experiment/2019-11-19-11_06_31/results/0805_x1_HR.png')
    file_path = 'C:/Users/guanghaoyin/Desktop/eccv2020/fig/VisualSOTA/Urban100/'
    # namel = file_path + 'LR_0001l.JPG'
    # namel = file_path + 'LR_0001o.JPG'
    namel = file_path + 'Our.png'
    # namel = file_path + '0858l.png'
    name = file_path + 'HR.png'
    img_hr = cv2.imread(name)
    img_sr = cv2.imread(namel)
    if use_Y:
        img_hr = bgr2ycbcr(img_hr)
        img_sr = bgr2ycbcr(img_sr)
        # name = file_path + 'yby.png'
        # cv2.imwrite(name,img_sr)
    ssim = cal_ssim(img_hr,img_sr)
    psnr = cal_psnr(img_hr,img_sr)
    print('psnr/ssim:  {:.2f}/{:.4f}'.format(psnr,ssim))
