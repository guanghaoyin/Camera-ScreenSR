import Data.utils as utils
import os
from Model.DURCAN_18 import DURCAN
# from Model.DURCAN_12_1 import DURCAN
from option import args
from Data.dataloader import MSDataLoader
from importlib import import_module
import torch
import torch.nn as nn
from tqdm import tqdm
from Metrics.cal_PSNR_SSIM import cal_psnr_tensor, cal_ssim_tensor,bgr2ycbcr,cal_ssim,cal_psnr
import imageio
import time
use_ycbcr = True
module_test = import_module('Data.' + args.data_test)
testset = getattr(module_test, args.data_test)(args,train = False)
loader_test = MSDataLoader(args,testset,batch_size=1,shuffle=False)
netSR = DURCAN(args)

if args.load_model == '.':
    raise NotImplementedError
else:
    if args.n_GPUs == 1:
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_best.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_12_l_comp_latest.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_12_comp_latest.pt')
        # netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_6_latest_finetune.pt')
        # netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_ablation_6_shallow_latest.pt')
        # netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_ablation_12_latest.pt')
        netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_ablation_18_latest.pt')
        # netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_12_comp_tanh_latest.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_latest_shallow_origin.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/Important_DURCAN_latest.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_color.pt')
        # netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_latest_deep.pt')
        netSR = netSR.to(torch.device('cuda:' + str(args.GPU_ID)))
    else:
        netSR = nn.DataParallel(netSR, range(args.n_GPUs)).cuda()
        netSR = utils.load_GPUS(netSR, args.load_model + '/model/Important_DURCAN_latest.pt')
        # netSR = utils.load_GPUS(netSR, args.load_model + '/model/DURCAN_latest_deep.pt')
        netSR = netSR.module

params = list(netSR.parameters())  # 所有参数放在params里
k = 0
# 3518.125000
for i in netSR.parameters():
    k += i.numel()
print('Model {} : params: {:4f}'.format(netSR._get_name(), k / 1000))

netSR.eval()
eval_psnr = 0
eval_ssim = 0
list_psnr = []
list_ssim = []
eval_t = 0
with torch.no_grad():
    for idx_img, (lr, hr, filename) in enumerate(tqdm(loader_test, ncols=80)):
        filename = filename[0]
        lr,hr = utils.tensor_prepare([lr, hr], args)
        start = time.time()
        SR = netSR(lr)
        torch.cuda.synchronize()
        end = time.time()
        t = end - start
        eval_t += t
        print('running time:%f' % t)
        SR = utils.unnormalize(SR)
        hr = utils.unnormalize(hr)
        #save_SR
        save_path = args.load_model + '/eval_result'
        SR = utils.quantize(SR, args.rgb_range)
        normalized_SR = SR[0].data.mul(255 / args.rgb_range)
        ndarr_SR = normalized_SR.byte().permute(1, 2, 0).cpu().numpy()
        normalized_HR = hr[0].data.mul(255 / args.rgb_range)
        ndarr_HR = normalized_HR.byte().permute(1, 2, 0).cpu().numpy()
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # imageio.imsave('{}/{}.png'.format(save_path,filename),ndarr_SR)
        #cal psnr
        if use_ycbcr:
            single_psnr = cal_psnr(bgr2ycbcr(ndarr_HR),bgr2ycbcr(ndarr_SR))
        else:
            single_psnr = cal_psnr_tensor(SR, hr, int(args.scale), args.rgb_range)
        list_psnr.append(single_psnr)
        eval_psnr += single_psnr
        print('\ttest img {} psnr: {}'.format(filename,single_psnr))
        #cal_ssim
        if use_ycbcr:
            single_ssim = cal_ssim(bgr2ycbcr(ndarr_HR), bgr2ycbcr(ndarr_SR))
        else:
            single_ssim = cal_ssim_tensor(SR,hr,args.rgb_range)
        list_ssim.append(single_ssim)
        eval_ssim += single_ssim
        print('\ttest img {} ssim: {}'.format(filename,single_ssim))

        torch.cuda.empty_cache()

ave_psnr = eval_psnr / len(loader_test)
ave_ssim = eval_ssim / len(loader_test)
ave_t = eval_t / len(loader_test)
print('\taverage psnr: %s'%ave_psnr)
print('\taverage ssim: %s'%ave_ssim)
print('\taverage time: %s'%ave_t)


with open(args.load_model + '/eval_result/list_psnr.txt', 'w') as f:
    f.write(str(list_psnr))
    f.write('ave_psnr:'+str(ave_psnr))
with open(args.load_model + '/eval_result/list_ssim.txt', 'w') as f:
    f.write(str(list_ssim))
    f.write('ave_ssim:'+str(ave_ssim))
