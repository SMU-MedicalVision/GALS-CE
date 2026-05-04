import os
from util.Nii_utils import NiiDataRead, NiiDataWrite
import numpy as np
from skimage import transform


taget_size = (32, 160, 192)
values_clip = (-55, 145)
data_dir = r'./Liver_CE_classifiy_data/'
save_dir = r'./Liver_CE_classifiy_data_preprocessed_before_mask'

for ID in sorted(os.listdir(data_dir)):
    print(len(os.listdir(os.path.join(save_dir, ID))))
    print(ID)
    os.makedirs(os.path.join(save_dir, ID), exist_ok=True)
    for mode in ['NC', 'AP', 'PVP', 'DP']:
        img, spacing, origin, direction = NiiDataRead(os.path.join(data_dir, ID, '{}_img.nii.gz'.format(mode)))
        mask_liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, '{}_liver_mask.nii.gz'.format(mode)))
        mask_tumor, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, '{}_Tumor_mask.nii.gz'.format(mode)))

        z_, x_, y_ = mask_liver.nonzero()
        z1 = z_.min()
        z2 = z_.max()
        x1 = x_.min()
        x2 = x_.max()
        y1 = y_.min()
        y2 = y_.max()

        img = img[z1: z2 + 1, x1: x2 + 1, y1:y2 + 1]
        mask_liver = mask_liver[z1: z2 + 1, x1: x2 + 1, y1:y2 + 1]
        mask_tumor = mask_tumor[z1: z2 + 1, x1: x2 + 1, y1:y2 + 1]

        img = np.clip(img, values_clip[0], values_clip[1])
        img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

        spacing_z = (spacing[0] * img.shape[0]) / taget_size[0]
        spacing_x = (spacing[1] * img.shape[1]) / taget_size[1]
        spacing_y = (spacing[2] * img.shape[2]) / taget_size[2]

        img = transform.resize(img, taget_size, order=0, mode='constant',
                               clip=False, preserve_range=True, anti_aliasing=False)
        mask_liver = transform.resize(mask_liver, taget_size, order=0, mode='constant',
                                     clip=False, preserve_range=True, anti_aliasing=False)
        mask_tumor = transform.resize(mask_tumor, taget_size, order=0, mode='constant',
                                      clip=False, preserve_range=True, anti_aliasing=False)
        NiiDataWrite(os.path.join(save_dir, ID, '{}_img.nii.gz'.format(mode)), img,
                     np.array([spacing_z, spacing_x, spacing_y]), origin, direction)
        NiiDataWrite(os.path.join(save_dir, ID, '{}_mask_liver.nii.gz'.format(mode)), mask_liver,
                     np.array([spacing_z, spacing_x, spacing_y]), origin, direction, as_type=np.uint8)
        NiiDataWrite(os.path.join(save_dir, ID, '{}_mask_tumor.nii.gz'.format(mode)), mask_tumor,
                     np.array([spacing_z, spacing_x, spacing_y]), origin, direction, as_type=np.uint8)