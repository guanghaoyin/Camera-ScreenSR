import torch
import torch.nn as nn
from torch.autograd import Variable
import Data.utils as utils
import Loss
from Loss import tv_loss
import Model
from Model.Net import DURCAN
from Data.option import args
from importlib import import_module
from Data.dataloader import MSDataLoader
from decimal import Decimal
from Loss import visual_loss
from tqdm import tqdm
from Metrics.cal_PSNR_SSIM import cal_psnr_tensor
import random

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)
loss = Loss.Loss(args,checkpoint)
module_train = import_module('Data.' + args.data_train)
module_test = import_module('Data.' + args.data_test)
trainset = getattr(module_train, args.data_train)(args)
testset = getattr(module_test, args.data_test)(args,train = False)

loader_train = MSDataLoader(args,trainset,batch_size=args.batch_size,shuffle=True)
loader_test = MSDataLoader(args,testset,batch_size=1,shuffle=False)

model = Model.Model(args,checkpoint)
model.load(args.load_model+ '/model/model_latest.pt')
model = model.to(torch.device('cuda:' + str(args.GPU_ID)))
netG = model.model.netG
# netG = model.model.module.netG
netD = model.model.netD
# netD = model.model.module.netD

netSR = DURCAN(args)
netSR.to(torch.device('cuda'))
if args.load_model == '.':
    netSR = netSR.to(torch.device('cuda:'+str(args.GPU_ID)))
    if not args.cpu and args.n_GPUs > 1:
        netSR = nn.DataParallel(netSR, range(args.n_GPUs)).cuda()
        netSR = netSR.module
else:
    if args.n_GPUs == 1:
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_best.pt')
        netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_latest.pt')
        # netSR = utils.load_GPU(netSR,args.load_model + '/model/DURCAN_latestX4.pt')
        netSR = netSR.to(torch.device('cuda:'+str(args.GPU_ID)))
    else:
        netSR = nn.DataParallel(netSR, range(args.n_GPUs)).cuda()
        netSR = utils.load_GPUS(netSR, args.load_model + '/model/DURCAN_best.pt')
        # netSR = utils.load_GPUS(netSR, args.load_model + '/model/DURCAN_latestX4.pt')
        netSR = netSR.module

optimizerSR = utils.make_optimizer(args, netSR)
schedulerSR = utils.make_scheduler(args, optimizerSR)


adversarial_criterion = nn.BCELoss()

print(netG)
print(netD)
print(netSR)

visualizer_env = 'DURCGAN' + '_' + args.data_train + '_' + args.loss
visualizer = visual_loss.Visualizer(env=visualizer_env)
Gloss_list = []
Dloss_list = []
SRloss_list = []
psnr_list = []
train_total_step = len(loader_train)
test_total_step = len(loader_test)
for epoch in range(args.epochs):
    loss.start_log()
    timer_data, timer_model = utils.timer(), utils.timer()
    schedulerSR.step()
    learning_rateSR = schedulerSR.get_lr()[0]
    checkpoint.write_log('[Epoch {}]\tLearning rateSR: {:.2e}'.format(epoch,Decimal(learning_rateSR)))
    for batch, (lr, hr, idx_scale) in enumerate(loader_train):
        netG.eval()
        netSR.train()
        lr, hr = utils.tensor_prepare([lr, hr],args)
        lr.hr = Variable(lr.cuda()),Variable(hr.cuda())
        timer_data.hold()
        timer_model.tic()
        optimizerSR.zero_grad()
        sr = netG(hr)
        if random.random() < 0.5:
            SR = netSR(sr)
        else:
            SR = netSR(lr)
        # sr *= args.rgb_range
        # sr = utils.quantize(sr, args.rgb_range)

        #update SR network
        netSR.zero_grad()
        # tvloss = TVLoss(SR) * (2e-9)
        SR_loss = loss(SR,hr)
        SR_loss_item = SR_loss.item()
        SRloss_list.append(SR_loss_item)
        SR_loss.backward()
        optimizerSR.step()

        timer_model.hold()

        print('Epoch [{}/{}],Step[{}/{}]'.format(epoch + 1, args.epochs, batch + 1, train_total_step))
        print('LossSR:{}'.format(SR_loss_item))

        timer_data.tic()
        visualizer.plot_many_stack({'SR_loss': SR_loss_item})

        if batch == 0:
            checkpoint.write_log('\nEvaluation:')
            # checkpoint.add_log(torch.zeros(1, len(args.scale)))
            print('start evaluation')
            netSR.eval()
            timer_test = utils.timer()
            with torch.no_grad():
                eval_psnr = 0
                for idx_img, (lr, hr, filename) in enumerate(tqdm(loader_test, ncols=80)):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = utils.tensor_prepare([lr, hr], args)
                    else:
                        lr = utils.tensor_prepare([lr], args)[0]

                    SR = netSR(lr)
                    # sr = utils.quantize(sr, args.rgb_range)
                    SR = utils.unnormalize(SR)
                    hr = utils.unnormalize(hr)
                    lr = utils.unnormalize(lr)
                    save_list = [sr, SR]
                    if not no_eval:
                        eval_psnr += cal_psnr_tensor(SR, hr, int(args.scale), args.rgb_range)
                        save_list.extend([lr, hr])
                    print('test step:[{}/{}]\n'.format(idx_img + 1, test_total_step))
                    if args.save_results:
                        checkpoint.save_results(epoch, filename, save_list, args.scale,postfix = ('SR', 'DUR_SR', 'LR', 'HR'))
                        netSR.save(checkpoint.dir, epoch)
                ave_psnr = eval_psnr / len(loader_test)
                checkpoint.log = ave_psnr
                checkpoint.write_log(
                    '[{} x{}]\tPSNR: {:.3f}'.format(
                        args.data_test,
                        args.scale,
                        checkpoint.log,
                    )
                )

                checkpoint.write_log(
                    'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
                )
                netSR.save(checkpoint.dir, epoch)
                psnr_list.append(ave_psnr)
                visualizer.plot_many_stack({visualizer_env + '_evaluate_psnr': ave_psnr})
                print('Evaluation psnr of the model is {}'.format(ave_psnr))
                torch.cuda.empty_cache()