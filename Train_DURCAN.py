import torch
import torch.nn as nn
from torch.autograd import Variable
import imageio
import Data.utils as utils
import Loss
from Model.Net import DURCAN
from option import args
from importlib import import_module
from Data.dataloader import MSDataLoader
from decimal import Decimal
from Loss import visual_loss
from tqdm import tqdm
from Metrics.cal_PSNR_SSIM import cal_psnr_tensor,cal_ssim_tensor

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)
loss = Loss.Loss(args,checkpoint)

module_train = import_module('Data.' + args.data_train)
module_test = import_module('Data.' + args.data_test)
trainset = getattr(module_train, args.data_train)(args)
testset = getattr(module_test, args.data_test)(args,train = False)

loader_train = MSDataLoader(args,trainset,batch_size=args.batch_size,shuffle=True)
loader_test = MSDataLoader(args,testset,batch_size=1,shuffle=False)
netSR = DURCAN(args)
# TVLoss = loss.tv_loss.TVLoss()
# TVLoss = TVLoss.to(torch.device('cuda:'+str(args.GPU_ID)))
if args.load_model == '.':
    netSR = netSR.to(torch.device('cuda:'+str(args.GPU_ID)))
    if not args.cpu and args.n_GPUs > 1:
        netSR = nn.DataParallel(netSR, range(args.n_GPUs)).cuda()
        netSR = netSR.module
else:
    if args.n_GPUs == 1:
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_best.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/Important_DURCAN_latest.pt')
        netSR = utils.load_GPU(netSR, args.load_model + '/model/DURCAN_12_1_comp_tanh_latest.pt')

        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_12_comp_latest.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_12_comp_tanh_latest.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_latestX4.pt')
        netSR = netSR.to(torch.device('cuda:'+str(args.GPU_ID)))
    else:
        netSR = nn.DataParallel(netSR, range(args.n_GPUs)).cuda()
        netSR = utils.load_GPUS(netSR, args.load_model + '/model/DURCAN_best.pt')
        # netSR = utils.load_GPUS(netSR, args.load_model + '/model/DURCAN_latestX4.pt')
        netSR = netSR.module

optimizerSR = utils.make_optimizer(args, netSR)
schedulerSR = utils.make_scheduler(args, optimizerSR)
print(netSR)

visualizer_env = 'DURCAN' + '_' + args.data_train + '_' + args.loss
if args.load_model != '.': visualizer_env += '_load'
visualizer = visual_loss.Visualizer(env=visualizer_env)

train_total_step = len(loader_train)
test_total_step = len(loader_test)
SRloss_list = []
PSNR_list = []
for epoch in range(args.epochs):
    loss.start_log()
    loss.step()
    timer_data, timer_model = utils.timer(), utils.timer()
    schedulerSR.step()
    learning_rateSR = schedulerSR.get_lr()[0]
    checkpoint.write_log('[Epoch {}]\tLearning rateSR: {:.2e}'.format(epoch, Decimal(learning_rateSR)))
    for batch, (lr, hr, idx_scale) in enumerate(loader_train):
        netSR.train()
        lr, hr = utils.tensor_prepare([lr, hr],args)
        lr.hr = Variable(lr.cuda()),Variable(hr.cuda())
        timer_data.hold()
        timer_model.tic()
        optimizerSR.zero_grad()
        SR = netSR(lr)
        # SR = utils.unnormalize(SR)
        # update SR network
        # netSR.zero_grad()
        # tvloss = TVLoss(SR)*(2e-9)
        # SR_loss = loss(SR, hr) + tvloss
        SR_loss = loss(SR, hr)
        error_last = loss.log[-1, -1]
        if SR_loss.item() < args.skip_threshold * error_last:
            SR_loss_item = SR_loss.item()
            SRloss_list.append(SR_loss_item)
            SR_loss.backward()
            optimizerSR.step()
            # tvloss.backward()
            print('Epoch [{}/{}],Step[{}/{}]'.format(epoch + 1, args.epochs, batch + 1, train_total_step))
            print('LossSR:{}'.format(SR_loss_item))
            visualizer.plot_many_stack({visualizer_env + '_SRloss': SR_loss_item})
        else:
            print('Skip this batch {}! (Loss: {})'.format(
                batch + 1, loss.item()
            ))
        timer_model.hold()


        # if batch == 20:
        if batch == 1:
                # and epoch%2 == 0:
            checkpoint.write_log('\nEvaluation:')
            # checkpoint.add_log(torch.zeros(1, len(args.scale)))
            print('start evaluation')
            netSR.eval()
            timer_test = utils.timer()
            with torch.no_grad():
                eval_psnr = 0
                eval_ssim = 0
                for idx_img, (lr, hr, filename) in enumerate(tqdm(loader_test, ncols=80)):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = utils.tensor_prepare([lr, hr], args)
                    else:
                        lr = utils.tensor_prepare([lr], args)[0]

                    SR = netSR(lr)
                    SR = utils.unnormalize(SR)
                    hr = utils.unnormalize(hr)
                    # SR = utils.quantize(SR,args.rgb_range)
                    save_list = [SR]
                    # save_list = []
                    single_PSNR = cal_psnr_tensor(SR, hr, int(args.scale), args.rgb_range)
                    single_SSIM = cal_ssim_tensor(SR, hr, args.rgb_range)

                    print('\ntest img {} ssim: {}'.format(filename, single_SSIM))
                    print('\ntest img {} pnsr: {}'.format(filename, single_PSNR))
                    if not no_eval:
                        eval_psnr += single_PSNR
                        eval_ssim += single_SSIM
                        # save_list.extend([lr, hr])
                    if args.save_results:
                            # and epoch%20 == 0:
                        checkpoint.save_results(epoch, filename, save_list, args.scale,postfix=('DUR_SR', 'LR', 'HR'))
                        netSR.save(checkpoint.dir, epoch)
                    print('test step:[{}/{}]\n'.format(idx_img + 1, test_total_step))
                ave_psnr = eval_psnr / len(loader_test)
                ave_ssim = eval_ssim / len(loader_test)
                # checkpoint.log = ave_psnr
                PSNR_list.append(ave_psnr)
                with open(checkpoint.dir + '/psnr_reford.txt', 'w') as f:
                    f.write(str(PSNR_list))
                checkpoint.write_log(
                    '[{} x{}]\tPSNR\tSSIM: {:.3f}'.format(
                        args.data_test,
                        args.scale,
                        ave_psnr,
                        ave_ssim,
                    )
                )

                checkpoint.write_log(
                    'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
                )
                visualizer.plot_many_stack({'evaluate_psnr': ave_psnr})
                visualizer.plot_many_stack({'evaluate_ssim': ave_ssim})
                print('Evaluation psnr of the model is {}'.format(ave_psnr))
                print('Evaluation ssim of the model is {}'.format(ave_ssim))
                torch.cuda.empty_cache()
        timer_data.tic()
        loss.end_log(len(loader_train))
