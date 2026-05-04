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
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchvision.utils import make_grid
from Nii_utils import Save_Parameter, setup_seed
from data.dataset_gen import DatasetFromFolder, inverser_norm_ct
from Networks.Networks_gen.pix2pix3d_model_adn import Pix2Pix3DModel, print_current_message, update_learning_rate, save_networks


def main(opt):
    if opt.volume.lower() == 'crop':
        train_set = DatasetFromFolder(opt, dataset='Train')
        val_set = DatasetFromFolder(opt, dataset='Val')
    # elif opt.volume.lower() == 'resize':
    #     train_set = CA_Dataset_harmonize_3D(opt, dataset="train", istrain=True)
    #     val_set = CA_Dataset_harmonize_3D(opt, dataset="val", istrain=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=8, batch_size=opt.batch_size*8, shuffle=False)


    train_size = len(train_dataloader)
    val_size = len(val_dataloader)
    print('The number of train images = %d' % train_size)
    print('The number of val images = %d' % val_size)

    opt.display_freq = int(train_size/opt.save_img_num)
    opt.print_freq = int(train_size/opt.print_freq_num)

    model = Pix2Pix3DModel(opt)


    if not opt.isTrain or opt.continue_train:
       load_networks(opt, model)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/train'))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/val'), flush_secs=4)

    best_MAE_A2B_val_liver = 1000
    best_MAE_A2B_val_all = 1000
    # best_SSIM = 0
    total_iters = 0
    print('training')

    Save_Parameter(opt)

    for epoch in range(opt.epoch_count, opt.max_epochs + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        image_num = 0
        epoch_train_MAE_B2A = []
        epoch_train_MAE_A2B = []
        # if epoch == 10:
        #     print('The network is considered fully trained after 10 epochs! ')
        #     best_MAE_A2B_val = 1000
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
                print_current_message(epoch, epoch_iter, train_size, lr, MAE_A2B, MAE_B2A)
                current_iter_display = epoch + i / train_size
                # train_writer.add_scalar('losses', losses, total_iters)
                train_writer.add_scalar('learning_rate', lr, total_iters)
                train_writer.add_scalar('MAE_A2B',  MAE_A2B, total_iters)
                train_writer.add_scalar('MAE_B2A',  MAE_B2A, total_iters)


            iter_data_time = time.time()
        train_writer.add_scalar('time',  (time.time() - epoch_start_time)/60, epoch)
        update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max)
        epoch_val_start_time = time.time()
        if epoch % 1 == 0:
            epoch_val_MAE_A2B_liver = []
            epoch_val_MAE_A2B_all = []
            count = 0
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    # count += 1
                    # if count > 2:
                    #     break
                    Real_B = data['B']
                    Real_A = data['A']
                    MASK = data[opt.sample_masktype]
                    pred_lh, CT_pred = model.netG.forward1(Real_A.cuda())
                    CT_pred = CT_pred.cpu()
                    CT_pred[CT_pred < -1] = -1
                    CT_pred[CT_pred > 1] = 1
                    # CT_pred[MASK == 0] = -1
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
            # message = 'epoch[%d/%d] val[%d/%d]: The MAE of %s is %.3f , the total MAE is %.3f' % (epoch, opt.max_epochs,
            #                                                                                       sub + 1,
            #                                                                                       len(image_filenames),
            #                                                                                       image_filenames[sub], MAE,
            #                                                                                       np.mean(epoch_val_MAE))
            if epoch_val_MAE_A2B_liver < best_MAE_A2B_val_liver:
                try:
                    os.remove(os.path.join(opt.model_results, f'best_MAE_val_liver_epoch{best_val_epoch_liver - 1}.pth'))
                except:
                    pass
                best_MAE_A2B_val_liver = epoch_val_MAE_A2B_liver
                save_networks(opt, f'best_MAE_val_liver_epoch{epoch}', model, epoch)
                best_val_epoch_liver = epoch
            if epoch_val_MAE_A2B_all < best_MAE_A2B_val_all:
                try:
                    os.remove(os.path.join(opt.model_results, f'best_MAE_val_all_epoch{best_val_epoch_all - 1}.pth'))
                except:
                    pass
                best_MAE_A2B_val_all = epoch_val_MAE_A2B_all
                save_networks(opt, f'best_MAE_val_all_epoch{epoch}', model, epoch)
                best_val_epoch_all = epoch

            # if epoch_val_SSIM > best_SSIM:
            #     best_SSIM = epoch_val_SSIM
            #     torch.save(model.netG.state_dict(), os.path.join(opt.model_results, 'best_SSIM.pth'))
        val_writer.add_scalar('best_MAE_A2B_val_liver', best_MAE_A2B_val_liver, epoch)
        val_writer.add_scalar('best_MAE_A2B_val_all', best_MAE_A2B_val_all, epoch)
        print('saving the model')
        save_networks(opt, f'latest_epoch{epoch}', model, epoch)
        try:
            os.remove(os.path.join(opt.model_results, f'latest_epoch{epoch-1}.pth'))
        except:
            pass


        print('End of epoch %d / %d \t Time Taken: %.1f min' % (epoch, opt.max_epochs, (time.time() - epoch_start_time)/60))

    train_writer.close()



if __name__ == '__main__':
    # root = '/home/zky/Github/GALS-CE/RAW_DATA/'
    # for ID in sorted(os.listdir(root)):
    #     if not os.path.isdir(os.path.join(root, ID)) or len(os.listdir(os.path.join(root, ID))) < 4:
    #         print(f"Skipping {ID} because it is not a directory or does not contain enough files.")
    #         continue
    #     os.rename(os.path.join(root, ID, 'Venous_Tumor_mask.nii.gz'), os.path.join(root, ID, 'Tumor_mask.nii.gz'))
    #     os.rename(os.path.join(root, ID, 'Liver_mask.nii.gz'), os.path.join(root, ID, 'Liver_mask.nii.gz'))
    #     # os.rename(os.path.join(root, ID, 'NC.nii.gz'), os.path.join(root, ID, 'NC.nii.gz'))
    #     os.rename(os.path.join(root, ID, 'AP.nii.gz'), os.path.join(root, ID, 'AP.nii.gz'))
    #     os.rename(os.path.join(root, ID, 'Venous.nii.gz'), os.path.join(root, ID, 'PVP.nii.gz'))
    #     os.rename(os.path.join(root, ID, 'DP.nii.gz'), os.path.join(root, ID, 'DP.nii.gz'))

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
    parser.add_argument('--volume', type=str, choices=['resize', 'crop'], help='volume type', default='crop')
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

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    os.environ['PYTHONHASHSEED'] = '8'

    opt.device = torch.device('cuda:0') if opt.gpu else torch.device('cpu')
    setup_seed(opt.seed)



    if opt.quick_test:
        opt.max_epoch = 10
        opt.disx = 100
    # -------------- Experiment naming & directory setup --------------
    if not opt.save_dir or not opt.inference_only:
        save_name = 'bs{}-norm_{}_{}_{}_{}-z{}_x{}_y{}-volume_{}'.format(opt.batch_size, opt.CT_min, opt.CT_mid1,
                                                                         opt.CT_mid2, opt.CT_max, opt.depthSize,
                                                                         opt.ImageSize_x, opt.ImageSize_y, opt.volume)
        opt.checkpoints_dir = os.path.join(f'./main/trained_models/GALS-CE_gen/pretrain/Swin_ADN_{opt.target_modal}', save_name, current_time)
        opt.display_results = os.path.join(opt.checkpoints_dir, 'image_results')
        opt.model_results = os.path.join(opt.checkpoints_dir, 'model_results')
        opt.file_name_txt = os.path.join(opt.checkpoints_dir, 'train_message.txt')
        opt.prediction_results = os.path.join(opt.checkpoints_dir, 'prediction_results_')
        # opt.pretrain_model_path = os.path.join(opt.code_dir, opt.loss_pre_dir)
        pathlib.Path(opt.display_results).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opt.model_results).mkdir(parents=True, exist_ok=True)

    if not opt.inference_only:
        trainer = main(opt)
        pred(opt, trainer)
    else:
        pred(opt)
    if not opt.inference_only:
        print("CBSI_gen Training Done")
        print("-------------------------------------------")
        if opt.quick_test:
            print(f"Attention !! Please use this command to carry out the next stage of the quick test:\n python ./main/train_GALS-CE_cla.py --quick_test --gen_save_dir {opt.save_dir}")
        else:
            print(f"Attention !! If you want to use the model trained in this session, Please use this command to carry out the next stage of training:\n python ./main/train_GALS-CE_cla.py --gen_save_dir {opt.save_dir}")
        print("-------------------------------------------")
    else:
        print("CBSI_gen Inference Done")


