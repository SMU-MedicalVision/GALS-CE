import os
# import time
import torch
# import random
import pathlib
from datetime import datetime
import argparse
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import SimpleITK as sitk
# from datetime import datetime
# from torch.autograd import Variable
from torch.utils.data import DataLoader
# # from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# from monai.utils import set_determinism
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchvision.utils import make_grid
# from util.util import *
# from data.NC2CE_dataset import *
# from networks.pix2pix_pre_adn_MMD_idt import *
# from data.dataset_resize_zky import CA_Dataset_harmonize_3D
# from util.Nii_utils import Save_Parameter, setup_seed, NiiDataRead, NiiDataWrite
from Nii_utils import Save_Parameter, setup_seed
from data.dataset_gen import DatasetFromFolder
from Networks.Networks_gen.pix2pix3d_model_adn import Pix2Pix3DModel, load_networks

def main(opt):
    if opt.volume.lower() == 'crop':
        train_set = DatasetFromFolder(opt, dataset='Train')
        val_set = DatasetFromFolder(opt, dataset='Val')
    # elif opt.volume.lower() == 'resize':
    #     train_set = CA_Dataset_harmonize_3D(opt, dataset="train", istrain=True, synthesis_mode='NC2CE')
    #     val_set = CA_Dataset_harmonize_3D(opt, dataset="val", istrain=True, synthesis_mode='NC2CE')

    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=min(8, opt.batch_size * 4), batch_size=opt.batch_size * 4, shuffle=False)

    train_size = len(train_dataloader)
    val_size = len(val_dataloader)  # get the number of images in the dataset.
    print('The number of train images = %d' % train_size)
    print('The number of val images = %d' % val_size)

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
    # best_SSIM = 0
    total_iters = 0
    count_test = 0
    Save_Parameter(opt)
    if opt.phase == 'train':
        print('training')
        for epoch in range(opt.epoch_count, opt.max_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            image_num = 0
            epoch_train_MAE = []
            count = 0
            model.train()
            for i, data in enumerate(tqdm(train_dataloader)):  # inner loop within one epoch
                # count += 1
                # if count > 3:
                #     break

                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += 1
                epoch_iter += 1
                model.set_input(data, sample_masktype=opt.sample_masktype)  # unpack data from dataset and apply preprocessing
                model.forward()  # calculate loss functions, get gradients, update networks weights

                Real_ct = inverser_norm_ct(model.real_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
                Fake_ct = inverser_norm_ct(model.fake_B.detach().cpu().numpy(), opt.CT_min, opt.CT_max,opt.CT_mid1,opt.CT_mid2,opt.Norm_tr1,opt.Norm_tr2)
                ct_mask = model.mask.detach().cpu().numpy()

                # MAE = (np.abs(Fake_ct - Real_ct) * ct_mask).sum() / ct_mask.sum()
                MAE = np.mean(np.abs(Fake_ct - Real_ct))
                # SSIM = compare_ssim(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=opt.CT_max - opt.CT_min)
                epoch_train_MAE.append(MAE)

                if total_iters % opt.display_freq == 0:
                    image_num += 1
                    display_current_results(opt, get_current_visuals_pre_ADN(model), epoch, image_num)
                    train_writer.add_image('A2B_realB', make_grid(torch.tensor(np.clip(Real_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                        normalize=True), total_iters)
                    train_writer.add_image('A2B_fakeB', make_grid(torch.tensor(np.clip(Fake_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                        normalize=True), total_iters)

                    # losses = get_current_losses(model)
                    lr = model.optimizer_G.param_groups[0]['lr']
                    print_current_message(epoch, epoch_iter, train_size, lr, MAE, MAE1)
                    current_iter_display = epoch + i / train_size
                    train_writer.add_scalar('learning_rate', lr, total_iters)
                    train_writer.add_scalar('train_MAE',  MAE, total_iters)
                    # for k, v in losses.items():
                    #     train_writer.add_scalar('%s' % k, v, total_iters)

                iter_data_time = time.time()
            # model.G_scheduler.step()
            # model.D_scheduler.step()
            train_writer.add_scalar('time', time.time()-epoch_start_time, epoch)
            train_writer.add_scalar('MAE', np.mean(epoch_train_MAE), epoch)
            update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max)
            print('saving the model')
            save_networks(opt, f'latest_epoch{epoch}', model, epoch)
            try:
                os.remove(os.path.join(opt.model_results, f'latest_epoch{epoch - 1}.pth'))
            except:
                pass

            # -----------------------验证集（微血管侵犯+中山+南方）-----------------------------
            epoch_val_MAE_all = []
            epoch_val_MAE_mask = []
            epoch_val_SSIM = []
            epoch_val_PSNR = []
            epoch_val_start_time = time.time()
            count = 0
            model.eval()
            with torch.no_grad():
                for i, DATA in enumerate(tqdm(val_dataloader)):  # inner loop within one epoch# calculate loss functions, get gradients, update networks weights
                    # count += 1
                    # if count > 3:
                    #     break
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

                    # data_range = max(Fake_ct[ct_mask > 0].max() - Fake_ct[ct_mask > 0].min(),
                    #                  Real_ct[ct_mask > 0].max() - Real_ct[ct_mask > 0].min())
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
            # raw = sitk.GetImageFromArray(Fake_ct[:, :, :].astype(np.float32))
            # sitk.WriteImage(raw, join(opt.checkpoints_dir, 'image_results', f'epoch{epoch}_Val_fakeB_{i}.nii.gz'))

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
            print(message)
            with open(opt.val_txt, 'a') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

            if epoch_val_MAE_mask < best_MAE_val_mask:
                if epoch < 20:
                    try:
                        os.remove(os.path.join(opt.model_results, f'best_MAE_mask_val_epoch{epoch - 1}.pth'))
                    except:
                        pass
                best_MAE_val_mask = epoch_val_MAE_mask
                save_networks(opt, f'best_MAE_mask_val_epoch{epoch}', model, epoch)
                best_val_epoch_mask = epoch
            if epoch_val_MAE_all < best_MAE_val_all:
                if epoch < 20:
                    try:
                        os.remove(os.path.join(opt.model_results, f'best_MAE_all_val_epoch{epoch - 1}.pth'))
                    except:
                        pass
                best_MAE_val_all = epoch_val_MAE_all
                save_networks(opt, f'best_MAE_all_val_epoch{epoch}', model, epoch)
                best_val_epoch_all = epoch
            val_writer.add_scalar('best_MAE_all', best_MAE_val_all, epoch)
            val_writer.add_scalar('best_MAE_mask', best_MAE_val_mask, epoch)
            if epoch % 10 != 0 and epoch != opt.max_epochs:
            # if epoch % 1 != 0 and epoch != opt.max_epochs:
                continue
            # -----------------------测试集（微血管侵犯+中山+南方）-----------------------------
            count_test += 1
            epoch_test_MAE_all = []
            epoch_test_MAE_mask = []
            epoch_test_SSIM = []
            epoch_test_PSNR = []
            epoch_test_start_time = time.time()
            count = 0
            model.eval()
            with torch.no_grad():
                for i, DATA in enumerate(tqdm(test_dataloader)):  # inner loop within one epoch# calculate loss functions, get gradients, update networks weights
                    # count += 1
                    # if count > 3:
                    #     break
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

                    # data_range = max(Fake_ct[ct_mask > 0].max() - Fake_ct[ct_mask > 0].min(),
                    #                  Real_ct[ct_mask > 0].max() - Real_ct[ct_mask > 0].min())
                    data_range = opt.CT_max-opt.CT_min

                    SSIM = structural_similarity(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=data_range)
                    PSNR = peak_signal_noise_ratio(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=data_range)
                    epoch_test_MAE_mask.append(MAE_mask)
                    epoch_test_MAE_all.append(MAE_all)
                    epoch_test_SSIM.append(SSIM)
                    epoch_test_PSNR.append(PSNR)
                if count_test == 1:
                    test_writer.add_image('realB', make_grid(
                        torch.tensor(np.clip(Real_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                        normalize=True), epoch)
                test_writer.add_image('fakeB', make_grid(
                    torch.tensor(np.clip(Fake_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), epoch)
            # raw = sitk.GetImageFromArray(Fake_ct[:, :, :].astype(np.float32))
            # sitk.WriteImage(raw, join(opt.checkpoints_dir, 'image_results', f'epoch{epoch}_test_fakeB_{i}.nii.gz'))

            epoch_test_MAE_all = np.mean(epoch_test_MAE_all)
            epoch_test_MAE_mask = np.mean(epoch_test_MAE_mask)
            epoch_test_SSIM = np.mean(epoch_test_SSIM)
            epoch_test_PSNR = np.mean(epoch_test_PSNR)
            test_writer.add_scalar('MAE_all', epoch_test_MAE_all, epoch)
            test_writer.add_scalar('MAE_mask', epoch_test_MAE_mask, epoch)
            test_writer.add_scalar('SSIM', epoch_test_SSIM, epoch)
            test_writer.add_scalar('PSNR', epoch_test_PSNR, epoch)
            test_writer.add_scalar('time', (time.time() - epoch_test_start_time)/60, epoch)

            message = 'epoch[%d/%d] test:The total MAE,SSIM,PSNR is %.3f, %.3f, %.3f.' % (
            epoch, opt.max_epochs, np.mean(epoch_test_MAE_mask),
            np.mean(epoch_test_SSIM),
            np.mean(epoch_test_PSNR))
            print(message)
            with open(opt.test_txt, 'a') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
            # -----------------------外部测试集（微血管侵犯+中山+南方）-----------------------------
            epoch_wtest_MAE_all = []
            epoch_wtest_MAE_mask = []
            epoch_wtest_SSIM = []
            epoch_wtest_PSNR = []
            epoch_wtest_start_time = time.time()
            count = 0
            model.eval()
            with torch.no_grad():
                for i, DATA in enumerate(tqdm(wtest_dataloader)):  # inner loop within one epoch# calculate loss functions, get gradients, update networks weights
                    # count += 1
                    # if count > 3:
                    #     break
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

                    # data_range = max(Fake_ct[ct_mask > 0].max() - Fake_ct[ct_mask > 0].min(),
                    #                  Real_ct[ct_mask > 0].max() - Real_ct[ct_mask > 0].min())
                    data_range = opt.CT_max-opt.CT_min

                    SSIM = structural_similarity(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=data_range)
                    PSNR = peak_signal_noise_ratio(Fake_ct[ct_mask > 0], Real_ct[ct_mask > 0], data_range=data_range)
                    epoch_wtest_MAE_mask.append(MAE_mask)
                    epoch_wtest_MAE_all.append(MAE_all)
                    epoch_wtest_SSIM.append(SSIM)
                    epoch_wtest_PSNR.append(PSNR)
                if count_test == 1:
                    wtest_writer.add_image('realB', make_grid(
                        torch.tensor(np.clip(Real_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                        normalize=True), epoch)
                wtest_writer.add_image('fakeB', make_grid(
                    torch.tensor(np.clip(Fake_ct[0, 0, 0::4] + 100, 0, 300) / 300 * 255).unsqueeze(1), 2,
                    normalize=True), epoch)
            # raw = sitk.GetImageFromArray(Fake_ct[:, :, :].astype(np.float32))
            # sitk.WriteImage(raw, join(opt.checkpoints_dir, 'image_results', f'epoch{epoch}_wtest_fakeB_{i}.nii.gz'))

            epoch_wtest_MAE_all = np.mean(epoch_wtest_MAE_all)
            epoch_wtest_MAE_mask = np.mean(epoch_wtest_MAE_mask)
            epoch_wtest_SSIM = np.mean(epoch_wtest_SSIM)
            epoch_wtest_PSNR = np.mean(epoch_wtest_PSNR)
            wtest_writer.add_scalar('MAE_all', epoch_wtest_MAE_all, epoch)
            wtest_writer.add_scalar('MAE_mask', epoch_wtest_MAE_mask, epoch)
            wtest_writer.add_scalar('SSIM', epoch_wtest_SSIM, epoch)
            wtest_writer.add_scalar('PSNR', epoch_wtest_PSNR, epoch)
            wtest_writer.add_scalar('time', (time.time() - epoch_wtest_start_time)/60, epoch)

            message = 'epoch[%d/%d] wtest:The total MAE,SSIM,PSNR is %.3f, %.3f, %.3f.' % (
            epoch, opt.max_epochs, np.mean(epoch_wtest_MAE_mask),
            np.mean(epoch_wtest_SSIM),
            np.mean(epoch_wtest_PSNR))
            print(message)
            with open(opt.wtest_txt, 'a') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')



if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='1', help='which gpu is used')
    parser.add_argument('--max_epochs', type=int, default=100, help='# max_epoch')
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
    parser.add_argument('--volume', type=str, choices=['resize', 'crop'], help='volume type', default='crop')
    parser.add_argument('--sample_masktype', type=str, choices=['body_mask', 'liver_mask'], default='liver_mask')
    parser.add_argument('--depthSize', type=int, default=8, help='depth for 3d images')
    parser.add_argument('--ImageSize_x', type=int, default=256, help='then crop to this size')
    parser.add_argument('--ImageSize_y', type=int, default=256, help='then crop to this size')
    parser.add_argument('--target_modal', type=str, choices=['AP', 'PVP', 'DP', 'all'], default='AP')
    parser.add_argument('--sample_Npatch', type=int, default=12)
    parser.add_argument('--sample_valuerange', type=tuple, default=(50, 150))

    # -------------------- Model settings
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--D_model', type=str, default='new_train', help='specify discriminator architecture [wave3DDiscriminator | n_layers | swinDiscriminator]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--G_model', type=str, default='Pre_Swin_ADN_10', help='specify generator architecture [global | global_trans | swin_trans ]')
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
    parser.add_argument('--lambda_idt', type=float, default=5, help='weight for identity  loss')
    parser.add_argument('--lambda_L0', type=float, default=5, help='weight for L1 loss')
    parser.add_argument('--lambda_NCC', type=float, default=10, help='weight for precetural loss')
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
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    os.environ['PYTHONHASHSEED'] = '8'

    opt.device = torch.device('cuda:0') if opt.gpu else torch.device('cpu')
    setup_seed(opt.seed)


    # -------------- Experiment naming & directory setup --------------
    if not opt.save_dir or not opt.inference_only:
        save_name = 'bs{}-norm_{}_{}_{}_{}-z{}_x{}_y{}-volume_{}'. format(opt.batch_size, opt.CT_min, opt.CT_mid1, opt.CT_mid2, opt.CT_max, opt.depthSize, opt.ImageSize_x, opt.ImageSize_y, opt.volume)
        opt.checkpoints_name = 'ours_%s_%s_modal%s' % (opt.G_model, opt.D_model, opt.target_modal)
        opt.checkpoints_dir = os.path.join('./main/trained_models/GALS-CE_gen/train', opt.checkpoints_name, save_name, current_time)
        opt.display_results = os.path.join(opt.checkpoints_dir, 'image_results')
        opt.model_results = os.path.join(opt.checkpoints_dir, 'model_results')
        # opt.file_name_txt = os.path.join(opt.checkpoints_dir, 'train_message.txt')

        opt.val_results = os.path.join(opt.checkpoints_dir, 'val_results')
        opt.val_txt = os.path.join(opt.val_results, 'val_message.txt')
        # opt.pretrain_model_path = os.path.join(opt.code_dir, opt.loss_pre_dir)


        pathlib.Path(opt.display_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt.model_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt.val_results).mkdir(parents=True, exist_ok=True)

    if not opt.inference_only:
        trainer = main(opt)
        pred(opt, trainer)
    else:
        pred(pt)