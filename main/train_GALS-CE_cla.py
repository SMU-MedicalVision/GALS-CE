import os
import time
import glob
import torch
import argparse
import numpy as np
from os.path import join
from fileinput import close
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

from Nii_utils import setup_seed, Save_Parameter
from models.Networks_gen.Networks_UNet_DDPM import Unet_class
from models.Networks_gen.Networks_DDPM_trainer import GaussianDiffusion, DDPM_Trainer
from models.Networks_gen.Networks_simple_UNet_DDPM import simple_Unet_for_Improved_DDPM_class
from models.Networks_gen.Validation_inference import Model_Validation_multitask, Model_Inference_multitask
from dataset.Dataset_gen import Dataset_harmonize_2D, Dataset_harmonize, Dataset_harmonize_inference



def main(opt):
    pass

def pred(opt, net=None):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_epoch', type=int, default=200, help='all_epochs')
    parser.add_argument('--lr_max', type=float, default=1e-5, help='max learning rate')
    parser.add_argument('--bs', type=int, default=2, help='training input batch size')
    parser.add_argument('--num_threads', type=int, default=8, help='# threads for loading dataset')

    # -------------------- Inference settings
    parser.add_argument('--val_batch_size', type=int, default=4, help='Val/Test batch size')
    parser.add_argument('--save_dir', type=str, default='', help='./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}')  # Path for saving model parameters

    # -------------------- Quick test settings
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')

    opt = parser.parse_args()
    # torch.cuda.is_available = lambda: False
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(opt.seed)
    print("GALS-CE_gen Start")
    if opt.quick_test:
        opt.max_epoch = 10
        opt.ref_timestep = 10
        opt.val_timestep_scale = 0.1
        opt.model_name = 'simple_unet_Improved_32_class'
    # -------------- Experiment naming & directory setup --------------
    if not opt.save_dir or not opt.inference_only:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_name = 'bs{}_epoch{}_gae{}_seed{}_class_seg'.format(opt.bs, opt.max_epoch, opt.gae, opt.seed)
        opt.save_dir = join(
            './main/trained_models/GALS-CE_gen/{}_{}_{}_condition_act_{}_{}_{}'.format(opt.objective, opt.model_name, opt.loss,
                                                                              opt.activate, save_name, current_time))
    os.makedirs(join(opt.save_dir, 'train_model'), exist_ok=True)

    if not opt.inference_only:
        net = main(opt)
        pred(opt, net)
    else:
        pred(opt)
    if not opt.inference_only:
        print("GALS-CE_gen Training Done")
        print("-------------------------------------------")
        if opt.quick_test:
            print(f"Attention !! Please use this command to carry out the next stage of the quick test:\n python ./main/train_GALS-CE_cla.py --quick_test --gen_save_dir {opt.save_dir}")
        else:
            print(f"Attention !! If you want to use the model trained in this session, Please use this command to carry out the next stage of training:\n python ./main/train_GALS-CE_cla.py --gen_save_dir {opt.save_dir}")
        print("-------------------------------------------")
    else:
        print("GALS-CE_gen Inference Done")