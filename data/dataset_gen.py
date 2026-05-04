# from os import listdir
# import random
# from sklearn import preprocessing
# from torchvision import transforms#
# from PIL import Image
import torch
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
# from torch.utils.data import DataLoader
# from skimage import transform
# import torchvision.transforms as transforms
# import pickle
# from copy import deepcopy

import os
from os.path import join
import torch.utils.data as data
from Nii_utils import NiiDataRead

def normalization_ct(data, min_value, max_value,mid1,mid2,tr1,tr2):
    if type(data) is not np.ndarray:
        data = data.numpy()
    nor_data = np.zeros((data.shape[0], data.shape[1], data.shape[2])).astype('float')
    data[data<min_value]= min_value
    data[data> max_value]= max_value
    # nor_data = (data - min_value) / (max_value - min_value)
    k1=tr1/(mid1 -min_value) #k=(b-a)/(max-min)   [min_value,mid1]
    k2 = (tr2-tr1) / (mid2-mid1) # [mid1,mid2]
    k3=(1-tr2)/(max_value-mid2) #[mid2,max_value]
    nor_data[data<mid1]= k1*(data[data<mid1]-min_value) #y=a+k(x-min)
    # nor_data[data<10]=nor
    nor_data[(data >=mid1)&(data<mid2)] =tr1+ k2 * (data[(data >= mid1)&(data<mid2)] -mid1)
    nor_data[data >= mid2] = tr2 + k3 * (data[data >= mid2] - mid2)
    nor_data=(nor_data-0.5)*2
    return nor_data


def inverser_norm_ct(data,min_value, max_value,mid1,mid2,tr1,tr2):
    data = (data+1)/2
    if isinstance(data, np.ndarray):
        nor_data = np.zeros_like(data).astype('float')
    else:
        nor_data = torch.zeros_like(data)

    k1=tr1/(mid1 -min_value) #k=(b-a)/(max-min)   [min_value,mid1]
    k2 = (tr2-tr1) / (mid2-mid1) # [mid1,mid2]
    k3=(1-tr2)/(max_value-mid2) #[mid2,max_value]
    nor_data[data < tr1]=data[data < tr1]/k1+min_value #(y-a)/k+min
    # nor_data[data<10]=nor
    nor_data[(data >= tr1)&(data<tr2)] =(data[(data >= tr1)&(data<tr2)]-tr1)/k2+mid1
    nor_data[data >= tr2] =(data[data >= tr2]-tr2)/k3+mid2
    return nor_data


def randomcrop_Npatch_2mask(crop_size, crop_Npatch, mri1, ct, ct_mask, liver_mask, b_ct0=None, dataset='train', this_index=None, crop_valuerange=None):
    this_frame = crop_size
    img = mri1
    if b_ct0 is None or crop_valuerange is None:
        non_zero_z, non_zero_x, non_zero_y = np.where(ct_mask == 1)
    else:
        non_zero_z, non_zero_x, non_zero_y = np.where((b_ct0 > crop_valuerange[0]) & (b_ct0 < crop_valuerange[1]))
    non_zero_num = non_zero_x.shape[0]
    # if non_zero_num < crop_Npatch:
    #    crop_Npatch = non_zero_num
    if dataset == 'train':
        patch_index = random.sample(range(0, non_zero_num), crop_Npatch)
    else:
        # patch_index = range(0, non_zero_num)[::int(non_zero_num/(crop_Npatch-1))]
        # patch_index = range(0, non_zero_num)[::(non_zero_num // (crop_Npatch - 1)-crop_Npatch)]
        patch_index = range(0, non_zero_num)[::(non_zero_num // crop_Npatch)]
        assert len(patch_index) == crop_Npatch or len(patch_index) == crop_Npatch+1, patch_index
        patch_index = [patch_index[this_index % crop_Npatch]]
        crop_Npatch = 1
        # print('patch_idx', patch_index)

    patch_mri1 = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_ct = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_mask = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_liver_mask = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    # patch_mask = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.int)

    for idx in range(crop_Npatch):
        z_med = non_zero_z[patch_index[idx]]
        x_med = non_zero_x[patch_index[idx]]
        y_med = non_zero_y[patch_index[idx]]
        z_frame_size = int(this_frame[0] / 2)
        x_frame_size = int(this_frame[1] / 2)
        y_frame_size = int(this_frame[2] / 2)

        if z_med < z_frame_size:
            z_this_min =  z_med
            z_this_max = z_frame_size * 2+ z_med
        elif z_med + z_frame_size > img.shape[0]:
            z_this_max = img.shape[0]-z_med-1+this_frame[0]
            z_this_min = z_this_max - this_frame[0]
        else:
            z_this_min = z_med - z_frame_size
            z_this_max = z_med + z_frame_size
        if x_med < x_frame_size:
            x_this_min = x_med
            x_this_max = x_med+x_frame_size * 2
        elif x_med + x_frame_size > img.shape[1]:
            x_this_max = img.shape[1]-x_med-1+this_frame[1]
            x_this_min = x_this_max - this_frame[1]
        else:
            x_this_min = x_med - x_frame_size
            x_this_max = x_med + x_frame_size

        if y_med < y_frame_size:
            y_this_min =y_med
            y_this_max =y_med+ y_frame_size * 2
        elif y_med + y_frame_size > img.shape[2]:
            y_this_max = img.shape[2]-y_med-1+this_frame[2]
            y_this_min = y_this_max - this_frame[2]
        else:
            y_this_min = y_med - y_frame_size
            y_this_max = y_med + y_frame_size

        patch_mri1[idx, :, :, :] = mri1[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]
        patch_ct[idx, :, :, :] = ct[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]
        patch_mask[idx, :, :, :] = ct_mask[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]
        patch_liver_mask[idx, :, :, :] = liver_mask[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]
    return np.ascontiguousarray(patch_mri1), np.ascontiguousarray(patch_ct), np.ascontiguousarray(patch_mask), np.ascontiguousarray(patch_liver_mask)


class DatasetFromFolder(data.Dataset):
    def __init__(self, opt, dataset='Train'):
        self.sample_masktype = opt.sample_masktype
        self.dataset = dataset
        self.target_modal = opt.target_modal
        self.image_dir = opt.image_dir
        self.CT_max = opt.CT_max
        self.CT_min = opt.CT_min
        self.CT_mid1=opt.CT_mid1
        self.CT_mid2 = opt.CT_mid2
        self.Norm_tr1=opt.Norm_tr1
        self.Norm_tr2 = opt.Norm_tr2

        self.image_filenames = sorted(os.listdir(join(opt.image_dir, self.dataset)))

        if opt.quick_test:
            self.image_filenames = self.image_filenames[:2]

        # if opt.debug:
        #     print('!!!!!!!!!! Attention  Just for debug, only use 4 samples for training and testing.  !!!!!!!!!!!!')
        #     self.image_filenames = self.image_filenames[:opt.inf_batch_size]
        #     print('!!!!!!!!!! Attention  Just for debug, only use 4 samples for training and testing.  !!!!!!!!!!!!')

        self.crop_size = [opt.depthSize, opt.ImageSize_x, opt.ImageSize_y]
        if self.dataset == 'train':
            self.crop_Npatch = opt.crop_Npatch
            self.all_patch_num = self.crop_Npatch * len(self.image_filenames)
        else:
            self.crop_Npatch = 1
            self.all_patch_num = len(self.image_filenames)
        if self.target_modal == 'all':
            self.image_filenames = self.image_filenames * 3
            self.image_filenames = sorted(self.image_filenames)

    def __getitem__(self, index):
        this_index = int(index // self.crop_Npatch)
        self.ran_num = 1
        if self.target_modal == 'all':
            a_mri1, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, self.dataset, self.image_filenames[this_index], ['AP', 'PVP', 'DP'][index % 3]+'.nii.gz'))
        else:
            a_mri1, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, self.dataset, self.image_filenames[this_index], f'{self.target_modal}.nii.gz'))
        b_ct, spacing1, origin1, direction1 = NiiDataRead(
            join(self.image_dir, self.dataset, self.image_filenames[this_index], 'NC.nii.gz'))
        b_mask, spacing, origin, direction = NiiDataRead(
            join(self.image_dir, self.dataset, self.image_filenames[this_index], 'Body_mask.nii.gz'))
        liver_mask, spacing, origin, direction = NiiDataRead(
            join(self.image_dir, self.dataset, self.image_filenames[this_index], 'Liver_mask.nii.gz'))

        ct_max = self.CT_max
        ct_min = self.CT_min
        CT_mid1=self.CT_mid1
        CT_mid2 = self.CT_mid2
        Norm_tr1=self.Norm_tr1
        Norm_tr2 =self.Norm_tr2
        b_ct[b_ct < ct_min] = ct_min
        b_ct[b_ct > ct_max] = ct_max
        b_ct[b_mask == 0] = ct_min
        a_mri1[a_mri1< ct_min] = ct_min
        a_mri1[a_mri1 > ct_max] = ct_max
        a_mri1[b_mask == 0] = ct_min
        a_mri1 = normalization_ct(a_mri1,ct_min, ct_max,CT_mid1,CT_mid2,Norm_tr1,Norm_tr2)
        b_ct = normalization_ct(b_ct, ct_min, ct_max,CT_mid1,CT_mid2,Norm_tr1,Norm_tr2)

        if self.dataset == 'train':
            # if self.sample_masktype == 'liver_mask':
            #     A_image, B_image, b_patch_mask = randomcrop_Npatch(self.crop_size, self.ran_num, a_mri1, b_ct, liver_mask, dataset=self.dataset)
            # else:
            #     A_image, B_image, b_patch_mask = randomcrop_Npatch(self.crop_size, self.ran_num, a_mri1, b_ct, b_mask, dataset=self.dataset)
            A_image, B_image, Body_Mask, Liver_Mask = randomcrop_Npatch_2mask(self.crop_size, self.ran_num, a_mri1, b_ct, b_mask, liver_mask, dataset=self.dataset)

        else:
            # liver_mask = b_mask
            # print('name', self.image_filenames[this_index], 'index', index)
            A_image, B_image, Body_Mask, Liver_Mask = randomcrop_Npatch_2mask(self.crop_size, self.crop_Npatch, a_mri1, b_ct, liver_mask, liver_mask, dataset=self.dataset, this_index=index)

        A_image = torch.tensor(A_image).float()
        B_image = torch.tensor(B_image).float()
        Body_Mask = torch.tensor(Body_Mask).float()
        Liver_Mask = torch.tensor(Liver_Mask).float()

        if self.dataset == 'train':
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                                      transforms.RandomHorizontalFlip(p1),
                                      transforms.RandomVerticalFlip(p2),
                                      # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                                      # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                           ])
            A_image = self.trans(A_image)
            B_image = self.trans(B_image)
            Body_Mask = self.trans(Body_Mask)
            Liver_Mask = self.trans(Liver_Mask)

        return {'A': A_image,
                'B': B_image,
                'body_mask': Body_Mask,
                'liver_mask': Liver_Mask,
                # 'tumor_mask': Tumor_Mask,
                'name': self.image_filenames[this_index]}

    def __len__(self):
        return self.all_patch_num





