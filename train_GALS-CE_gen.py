import warnings
warnings.filterwarnings("ignore")
import os
import time
import torch
# import random
import pathlib
from datetime import datetime
import argparse
import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from torch.autograd import Variable
from torch.utils.data import DataLoader
# # from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# from monai.utils import set_determinism
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchvision.utils import make_grid
from Nii_utils import Save_Parameter, setup_seed
from data.dataset_gen import DatasetFromFolder, inverser_norm_ct
from Networks.Networks_gen.pix2pix3d_model_adn import *


def pretrain(opt):
    train_start_time = time.time()
    train_set = DatasetFromFolder(opt, dataset='Train')
    val_set = DatasetFromFolder(opt, dataset='Val')
    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=8, batch_size=opt.batch_size*8, shuffle=False)


    train_size = len(train_dataloader)
    val_size = len(val_dataloader)

    opt.display_freq = int(train_size/opt.save_img_num)
    opt.print_freq = int(train_size/opt.print_freq_num)

    model = Pix2Pix3DModel_pretrain(opt)


    if not opt.isTrain or opt.continue_train:
       load_networks(opt, model)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/train'))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/val'), flush_secs=4)

    best_MAE_A2B_val_liver = 1000
    best_MAE_A2B_val_all = 1000
    total_iters = 0

    Save_Parameter(opt)

    for epoch in range(opt.epoch_count, opt.max_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        image_num = 0
        epoch_train_MAE_B2A = []
        epoch_train_MAE_A2B = []
        pathlib.Path(opt.display_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt.model_results).mkdir(parents=True, exist_ok=True)
        count = 0
        for i, data in enumerate(train_dataloader):
            count += 1
            iter_start_time = time.time()
            total_iters += 1
            epoch_iter += 1
            model.set_input(data, opt.sample_masktype)
            model.forward(epoch)

            Real_ct = inverser_norm_ct(model.real_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
            Fake_ct = inverser_norm_ct(model.pred_ll.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
            ct_mask = model.mask.detach().cpu().numpy()
            MAE_A2B = np.mean(np.abs(Fake_ct - Real_ct))
            epoch_train_MAE_A2B.append(MAE_A2B)
            if total_iters % opt.display_freq == 0:
                train_writer.add_image('A2B_realB', make_grid(torch.tensor(np.clip(Real_ct[0, 0, 0::10]+100, 0, 300)/300 * 255).unsqueeze(1), 2, normalize=True), total_iters)
                train_writer.add_image('A2B_fakeB', make_grid(torch.tensor(np.clip(Fake_ct[0, 0, 0::10]+100, 0, 300)/300 * 255).unsqueeze(1), 2, normalize=True), total_iters)

            Real_ct = inverser_norm_ct(model.real_A.detach().cpu().numpy(), opt.CT_min, opt.CT_max, opt.CT_mid1, opt.CT_mid2, opt.Norm_tr1, opt.Norm_tr2)
            Fake_ct = inverser_norm_ct(model.pred_hh.detach().cpu().numpy(), opt.CT_min, opt.CT_max, opt.CT_mid1, opt.CT_mid2, opt.Norm_tr1, opt.Norm_tr2)
            MAE_B2A = np.mean(np.abs(Fake_ct - Real_ct))
            epoch_train_MAE_B2A.append(MAE_B2A)
            if total_iters % opt.display_freq == 0:
                train_writer.add_image('B2A_realA', make_grid(torch.tensor(np.clip(Real_ct[0, 0, 0::10]+100, 0, 300)/300 * 255).unsqueeze(1), 2, normalize=True), total_iters)
                train_writer.add_image('B2A_fakeA', make_grid(torch.tensor(np.clip(Fake_ct[0, 0, 0::10]+100, 0, 300)/300 * 255).unsqueeze(1), 2, normalize=True), total_iters)

                image_num += 1
                lr = model.optimizer_G.param_groups[0]['lr']
                current_iter_display = epoch + i / train_size
                # train_writer.add_scalar('losses', losses, total_iters)
                train_writer.add_scalar('learning_rate', lr, total_iters)
                train_writer.add_scalar('MAE_A2B',  MAE_A2B, total_iters)
                train_writer.add_scalar('MAE_B2A',  MAE_B2A, total_iters)



        train_writer.add_scalar('time',  (time.time() - epoch_start_time)/60, epoch)
        update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max)
        epoch_val_start_time = time.time()
        if epoch % 1 == 0:
            epoch_val_MAE_A2B_liver = []
            epoch_val_MAE_A2B_all = []
            count = 0
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    Real_B = data['B']
                    Real_A = data['A']
                    MASK = data[opt.sample_masktype]
                    pred_lh, CT_pred = model.netG.forward1(Real_A.cuda())
                    CT_pred = CT_pred.cpu()
                    CT_pred[CT_pred < -1] = -1
                    CT_pred[CT_pred > 1] = 1
                    CT_pred = inverser_norm_ct(CT_pred, opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
                    Real_B = inverser_norm_ct(Real_B, opt.CT_min, opt.CT_max, opt.CT_mid1, opt.CT_mid2, opt.Norm_tr1, opt.Norm_tr2)
                    MAE_A2B_liver = (np.abs(CT_pred - Real_B) * MASK).sum() / MASK.sum()
                    MAE_A2B_all = torch.mean(np.abs(CT_pred - Real_B))
                    epoch_val_MAE_A2B_all.append(MAE_A2B_all)
                    epoch_val_MAE_A2B_liver.append(MAE_A2B_liver)


                val_writer.add_image('A2B_realA', make_grid(
                    torch.tensor(np.clip(Real_A[0, 0, 0::10] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), epoch)
                val_writer.add_image('A2B_realB', make_grid(
                    torch.tensor(np.clip(Real_B[0, 0, 0::10] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), epoch)
                val_writer.add_image('A2B_fakeB', make_grid(
                    torch.tensor(np.clip(CT_pred[0, 0, 0::10] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), epoch)

            epoch_val_MAE_A2B_all = np.mean(epoch_val_MAE_A2B_all)
            epoch_val_MAE_A2B_liver = np.mean(epoch_val_MAE_A2B_liver)
            val_writer.add_scalar('MAE_A2B_liver', epoch_val_MAE_A2B_liver, epoch)
            val_writer.add_scalar('MAE_A2B_all', epoch_val_MAE_A2B_all, epoch)
            val_writer.add_scalar('time', (time.time() - epoch_val_start_time)/60, epoch)
            if epoch_val_MAE_A2B_liver < best_MAE_A2B_val_liver:
                try:
                    os.remove(os.path.join(opt.model_results, f'best_MAE_val_liver_epoch{best_val_epoch_liver}.pth'))
                except:
                    pass
                best_MAE_A2B_val_liver = epoch_val_MAE_A2B_liver
                save_networks(opt, f'best_MAE_val_liver_epoch{epoch}', model, epoch)
                best_val_epoch_liver = epoch
            if epoch_val_MAE_A2B_all < best_MAE_A2B_val_all:
                try:
                    os.remove(os.path.join(opt.model_results, f'best_MAE_val_all_epoch{best_val_epoch_all}.pth'))
                except:
                    pass
                best_MAE_A2B_val_all = epoch_val_MAE_A2B_all
                save_networks(opt, f'best_MAE_val_all_epoch{epoch}', model, epoch)
                best_val_epoch_all = epoch
        val_writer.add_scalar('best_MAE_A2B_val_liver', best_MAE_A2B_val_liver, epoch)
        val_writer.add_scalar('best_MAE_A2B_val_all', best_MAE_A2B_val_all, epoch)
        save_networks(opt, f'latest_epoch{epoch}', model, epoch)
        try:
            os.remove(os.path.join(opt.model_results, f'latest_epoch{epoch-1}.pth'))
        except:
            pass

    print('End of epoch %d / %d \t Time Taken: %.1f min' % (epoch, opt.max_epochs, (time.time() - train_start_time)/60))

    train_writer.close()
    val_writer.close()
    return os.path.join(opt.model_results, f'best_MAE_val_liver_epoch{best_val_epoch_liver}.pth')


def main(opt):
    train_start_time = time.time()
    train_set = DatasetFromFolder(opt, dataset='Train')
    val_set = DatasetFromFolder(opt, dataset='Val')

    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=min(8, opt.batch_size * 4), batch_size=opt.batch_size * 4, shuffle=False)

    train_size = len(train_dataloader)
    val_size = len(val_dataloader)

    opt.display_freq = train_size//opt.save_img_num
    opt.print_freq = train_size//opt.print_freq_num

    model = Pix2Pix3DModel(opt)

    if not opt.isTrain or opt.continue_train:
       load_networks(opt, model)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/train'))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/val'), flush_secs=4)

    best_MAE_val_mask = 1000
    best_MAE_val_all = 1000
    MAE1 = 1000
    total_iters = 0
    count_test = 0
    Save_Parameter(opt)


    for epoch in range(opt.epoch_count, opt.max_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0
        image_num = 0
        epoch_train_MAE = []
        count = 0
        model.train()
        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1
            epoch_iter += 1
            model.set_input(data, sample_masktype=opt.sample_masktype)  # unpack data from dataset and apply preprocessing
            model.forward()  # calculate loss functions, get gradients, update networks weights

            Real_ct = inverser_norm_ct(model.real_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
            Fake_ct = inverser_norm_ct(model.fake_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
            ct_mask = model.mask.detach().cpu().numpy()

            MAE = np.mean(np.abs(Fake_ct - Real_ct))
            epoch_train_MAE.append(MAE)

            if total_iters % opt.display_freq == 0:
                image_num += 1
                train_writer.add_image('A2B_realB', make_grid(torch.tensor(np.clip(Real_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), total_iters)
                train_writer.add_image('A2B_fakeB', make_grid(torch.tensor(np.clip(Fake_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), total_iters)


                lr = model.optimizer_G.param_groups[0]['lr']
                current_iter_display = epoch + i / train_size
                train_writer.add_scalar('learning_rate', lr, total_iters)
                train_writer.add_scalar('train_MAE',  MAE, total_iters)


        train_writer.add_scalar('time', time.time()-epoch_start_time, epoch)
        train_writer.add_scalar('MAE', np.mean(epoch_train_MAE), epoch)
        update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max)

        save_networks(opt, f'latest_epoch{epoch}', model, epoch)
        try:
            os.remove(os.path.join(opt.model_results, f'latest_epoch{epoch - 1}.pth'))
        except:
            pass

        # -----------------------val
        epoch_val_MAE_all = []
        epoch_val_MAE_mask = []
        epoch_val_SSIM = []
        epoch_val_PSNR = []
        epoch_val_start_time = time.time()
        count = 0
        model.eval()
        with torch.no_grad():
            for i, DATA in enumerate(val_dataloader):

                Real_B = DATA['B'].cuda()
                Fake_B = model.netG.forward(DATA['A'].cuda())
                ct_mask = DATA[opt.sample_masktype]
                Fake_B[Fake_B < -1] = -1
                Fake_B[Fake_B > 1] = 1

                Real_ct = inverser_norm_ct(Real_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
                Fake_ct = inverser_norm_ct(Fake_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
                ct_mask = ct_mask.numpy()


                MAE_mask = (np.abs(Fake_ct - Real_ct) * ct_mask).sum() / ct_mask.sum()
                MAE_all = np.mean(np.abs(Fake_ct - Real_ct))

                data_range = opt.CT_max-opt.CT_min

                SSIM = structural_similarity(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=data_range)
                PSNR = peak_signal_noise_ratio(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=data_range)
                epoch_val_MAE_mask.append(MAE_mask)
                epoch_val_MAE_all.append(MAE_all)
                epoch_val_SSIM.append(SSIM)
                epoch_val_PSNR.append(PSNR)
            if epoch == opt.epoch_count:
                val_writer.add_image('realB', make_grid(
                    torch.tensor(np.clip(Real_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), epoch)
            val_writer.add_image('fakeB', make_grid(
                torch.tensor(np.clip(Fake_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                normalize=True), epoch)

        epoch_val_MAE_all = np.mean(epoch_val_MAE_all)
        epoch_val_MAE_mask = np.mean(epoch_val_MAE_mask)
        epoch_val_SSIM = np.mean(epoch_val_SSIM)
        epoch_val_PSNR = np.mean(epoch_val_PSNR)
        val_writer.add_scalar('MAE_all', epoch_val_MAE_all, epoch)
        val_writer.add_scalar('MAE_mask', epoch_val_MAE_mask, epoch)
        val_writer.add_scalar('SSIM', epoch_val_SSIM, epoch)
        val_writer.add_scalar('PSNR', epoch_val_PSNR, epoch)
        val_writer.add_scalar('time', (time.time() - epoch_val_start_time)/60, epoch)

        message = 'epoch[%d/%d] val:The total MAE,SSIM,PSNR is %.3f, %.3f, %.3f.' % (
        epoch, opt.max_epochs, np.mean(epoch_val_MAE_mask),
        np.mean(epoch_val_SSIM),
        np.mean(epoch_val_PSNR))
        # print(message)
        with open(opt.val_txt, 'a') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        if epoch_val_MAE_mask < best_MAE_val_mask:
            try:
                os.remove(os.path.join(opt.model_results, f'best_MAE_mask_val_epoch{best_MAE_val_mask}.pth'))
            except:
                pass
            best_MAE_val_mask = epoch_val_MAE_mask
            save_networks(opt, f'best_MAE_mask_val_epoch{epoch}', model, epoch)
            best_val_epoch_mask = epoch
        if epoch_val_MAE_all < best_MAE_val_all:
            try:
                os.remove(os.path.join(opt.model_results, f'best_MAE_all_val_epoch{best_MAE_val_all}.pth'))
            except:
                pass
            best_MAE_val_all = epoch_val_MAE_all
            save_networks(opt, f'best_MAE_all_val_epoch{epoch}', model, epoch)
            best_val_epoch_all = epoch
        val_writer.add_scalar('best_MAE_all', best_MAE_val_all, epoch)
        val_writer.add_scalar('best_MAE_mask', best_MAE_val_mask, epoch)
        if epoch % 10 != 0 and epoch != opt.max_epochs:
            continue


    print('End of epoch %d / %d \t Time Taken: %.1f min' % (epoch, opt.max_epochs, (time.time() - train_start_time) / 60))
    train_writer.close()
    val_writer.close()
    return os.path.join(opt.model_results, f'best_MAE_mask_val_epoch{best_val_epoch_mask}.pth')


def pred(opt, model_paths_dict, model):
    for phase in ['AP', 'PVP', 'DP']:
        print(model_paths_dict[phase])
        with torch.no_grad():
            wait_list = [ID for ID in os.listdir(opt.image_dir) if ID in dataset_split][::-1]
            for i, ID in enumerate(tqdm(wait_list)):
                if os.path.exists(
                        join(opt.train_parameter_root, f'image_prediction_{opt.load_name}',
                             dataset, ID, f'{opt.target_modal}.nii.gz')):
                    print(f'Skip {ID}, already exists')
                    continue
                count += 1
                NC, spacing, origin, direction = NiiDataRead(join(opt.image_dir, ID, 'NC.nii.gz'))
                NC_norm = normalization_ct(NC, opt.CT_min, opt.CT_max, opt.CT_mid1, opt.CT_mid2, opt.Norm_tr1, opt.Norm_tr2)
                b_mask, _, _, _ = NiiDataRead(join(opt.image_dir, ID, 'Body_mask.nii.gz'))

                MASK_sample = np.zeros_like(b_mask)
                dis = 10120 * 2
                z, x, y = np.where(b_mask == 1)
                for num in range(len(x)):
                    if num % dis == 0:
                        deep = z[num]
                        height = y[num]
                        width = x[num]
                        MASK_sample[deep, height, width] = 1
                pred_image = test_pred(model, NC_norm[np.newaxis, ...], b_mask, opt, MASK_sample=MASK_sample)
                pred_image = inverser_norm_ct(pred_image, opt.CT_min, opt.CT_max, opt.CT_mid1, opt.CT_mid2,
                                              opt.Norm_tr1, opt.Norm_tr2)
                pred_image[b_mask == 0] = NC[b_mask == 0]

                os.makedirs(join(opt.train_parameter_root, f'image_prediction_{opt.load_name}', dataset, ID), exist_ok=True)
                NiiDataWrite(join(opt.train_parameter_root, f'image_prediction_{opt.load_name}', dataset, ID, f'{opt.target_modal}.nii.gz'), pred_image, spacing, origin, direction)


if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--max_epochs', type=int, default=50, help='# max_epoch')
    parser.add_argument('--batch_size', type=int, default=3, help='input batch size')
    parser.add_argument('--num_threads', type=int, default=3, help='# threads for loading data')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--isTrain', action='store_false', help='isTrain')
    parser.add_argument('--continue_train', default=False, help='continue training: load the latest model')
    parser.add_argument('--load_path_pretrain', type=str, default=None)
    parser.add_argument('--load_name', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

    parser.add_argument('--save_img_num', type=int, default=1, help='save_img_num')
    parser.add_argument('--print_freq_num', type=int, default=4, help='frequency of showing training results on console')
    # ----------------------------- path ----------------------------- #
    parser.add_argument('--image_dir', type=str, default="./RAW_DATA", help="name of the dataset")

    # -------------------- Data settings
    parser.add_argument('--CT_max', type=float, default=1000, help='CT_max')
    parser.add_argument('--CT_min', type=float, default=-1000, help='CT_min')
    parser.add_argument('--CT_mid1', type=float, default=-100, help='CT_mid1')
    parser.add_argument('--CT_mid2', type=float, default=300, help='CT_mid2')
    parser.add_argument('--Norm_tr1', type=float, default=0.2, help='Norm_tr1')
    parser.add_argument('--Norm_tr2', type=float, default=0.9, help='Norm_tr2')
    parser.add_argument('--sample_masktype', type=str, choices=['body_mask', 'liver_mask'], default='liver_mask')
    parser.add_argument('--depthSize', type=int, default=8, help='depth for 3d images')
    parser.add_argument('--ImageSize_x', type=int, default=256, help='then crop to this size')
    parser.add_argument('--ImageSize_y', type=int, default=256, help='then crop to this size')
    parser.add_argument('--target_modal', type=str, choices=['AP', 'PVP', 'DP', 'all'], default='all')
    parser.add_argument('--sample_Npatch', type=int, default=None)

    # -------------------- Model settings
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--D_model', type=str, default='basic', help='specify discriminator architecture [wave3DDiscriminator | n_layers | swinDiscriminator]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--G_model', type=str, default='Swin_ADN', help='specify generator architecture [global | global_trans | swin_trans ]')
    parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=16, help='# of discrim filters in the first conv layer')
    parser.add_argument('--num_down', type=int, default=3, help='# of gen encoder downsampling')
    parser.add_argument('--num_up', type=int, default=3, help='# of gen decoder upsampling')
    parser.add_argument('--n_downsampling', type=int, default=3, help='# of gen transformer')
    parser.add_argument('--n_layers_D', type=int, default=2, help='only used if netD==n_layers')
    parser.add_argument('--G_norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--D_norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='networks initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

    # -------------------- Loss function
    parser.add_argument('--lambda_L1', type=float, default=20, help='weight for L1 loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr_max', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--l1_weight_inmask', type=float, help='[0,1]', default=0.8)
    parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--loss_pre_dir', type=str, default='perceive_loss/vgg19-dcbb9e9d.pth', help='resnet18_pretrain_path')

    # -------------------- Inference settings
    parser.add_argument('--val_bs', type=int, default=4, help='Val/Test batch size')
    parser.add_argument('--disx', type=int, default=10000, help='frequency of showing training results on console')  # 8000
    parser.add_argument('--save_dir', type=str, default='', help='./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}')  # Path for saving model parameters

    # -------------------- Quick test settings
    # parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')


    parser.add_argument('--quick_test', type=bool, default=True)

    opt_pre = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt_pre.gpu
    os.environ['PYTHONHASHSEED'] = '8'

    opt_pre.device = torch.device('cuda:0') if opt_pre.gpu else torch.device('cpu')
    setup_seed(opt_pre.seed)

    if opt_pre.quick_test:
        opt_pre.max_epochs = 10
        opt_pre.disx = 100
    # -------------- Experiment naming & directory setup --------------
    if not opt_pre.save_dir or not opt_pre.inference_only:
        save_name = 'bs{}-norm_{}_{}_{}_{}-z{}_x{}_y{}'.format(opt_pre.batch_size, opt_pre.CT_min, opt_pre.CT_mid1,
                                                                         opt_pre.CT_mid2, opt_pre.CT_max, opt_pre.depthSize,
                                                                         opt_pre.ImageSize_x, opt_pre.ImageSize_y)
        opt_pre.checkpoints_dir = os.path.join(f'./main/trained_models/GALS-CE_gen/pretrain/Swin_ADN_{opt_pre.target_modal}', save_name, current_time)
        opt_pre.display_results = os.path.join(opt_pre.checkpoints_dir, 'image_results')
        opt_pre.model_results = os.path.join(opt_pre.checkpoints_dir, 'model_results')
        opt_pre.file_name_txt = os.path.join(opt_pre.checkpoints_dir, 'train_message.txt')
        opt_pre.prediction_results = os.path.join(opt_pre.checkpoints_dir, 'prediction_results_')
        pathlib.Path(opt_pre.display_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt_pre.model_results).mkdir(parents=True, exist_ok=True)

    if not opt_pre.inference_only:
        print('-' * 50, "\nCBSI_gen Pretraining Start\n", '-' * 50)
        pretrain_model_path = pretrain(opt_pre)
        print('-' * 50, "\nCBSI_gen Pretraining Done\n", '-' * 50)


    # ----------------------------------
    opt_train = argparse.Namespace(**vars(opt_pre))
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='1', help='which gpu is used')
    parser.add_argument('--max_epochs', type=int, default=100, help='# max_epoch')
    # parser.add_argument('--target_modal', type=str, choices=['AP', 'PVP', 'DP', 'all'], default='AP')
    parser.add_argument('--sample_Npatch', type=int, default=12)
    parser.add_argument('--sample_valuerange', type=tuple, default=(50, 150))
    parser.add_argument('--D_model', type=str, default='new_train',
                        help='specify discriminator architecture [wave3DDiscriminator | n_layers | swinDiscriminator]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--G_model', type=str, default='Pre_Swin_ADN_10',
                        help='specify generator architecture [global | global_trans | swin_trans ]')
    parser.add_argument('--lambda_idt', type=float, default=5, help='weight for identity  loss')
    parser.add_argument('--lambda_L0', type=float, default=5, help='weight for L1 loss')
    parser.add_argument('--lambda_NCC', type=float, default=10, help='weight for precetural loss')
    opt_new = parser.parse_args()
    opt_train.__dict__.update(vars(opt_new))
    if opt_train.quick_test:
        opt_train.max_epochs = 10
        opt_train.disx = 100
    opt_train.load_path_pretrain = pretrain_model_path
    # -------------- Experiment naming & directory setup --------------
    if not opt_train.save_dir or not opt_train.inference_only:
        save_name = 'bs{}-norm_{}_{}_{}_{}-z{}_x{}_y{}'. format(opt_train.batch_size, opt_train.CT_min, opt_train.CT_mid1, opt_train.CT_mid2, opt_train.CT_max, opt_train.depthSize, opt_train.ImageSize_x, opt_train.ImageSize_y)
        opt_train.checkpoints_name = 'ours_%s_%s_modal%s' % (opt_train.G_model, opt_train.D_model, opt_train.target_modal)
        opt_train.checkpoints_dir = os.path.join('./main/trained_models/GALS-CE_gen/train', opt_train.checkpoints_name, save_name, current_time)
        opt_train.display_results = os.path.join(opt_train.checkpoints_dir, 'image_results')
        opt_train.model_results = os.path.join(opt_train.checkpoints_dir, 'model_results')

        opt_train.val_results = os.path.join(opt_train.checkpoints_dir, 'val_results')
        opt_train.val_txt = os.path.join(opt_train.val_results, 'val_message.txt')


        pathlib.Path(opt_train.display_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt_train.model_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt_train.val_results).mkdir(parents=True, exist_ok=True)

    if not opt_train.inference_only:
        print('-' * 50, "\nCBSI_gen Training Start\n", '-' * 50)
        model_paths_dict = {}
        for phase in ['AP', 'PVP', 'DP']:
            opt_train.target_modal = phase
            train_model_path = main(opt_train)
            print('-' * 50, f"\n{phase}_CBSI_gen Training Done\n", '-' * 50)
            model_paths_dict[phase] = train_model_path
        print('-' * 50, "\nCBSI_gen Inference Start\n", '-' * 50)
        pred(opt_train, model_paths_dict)
        print('-' * 50, "\nCBSI_gen Inference Done\n", '-' * 50)
    else:
        print('-' * 50, "\nCBSI_gen Inference Start\n", '-' * 50)
        model_paths_dict = {}
        model_paths_dict[phase] = train_model_path
        pred(opt_train, model_paths_dict)
        print('-' * 50, "\nCBSI_gen Inference Done\n", '-' * 50)





