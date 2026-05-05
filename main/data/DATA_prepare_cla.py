import os
import numpy as np
from os.path import join
from preprocess
from Nii_utils import NiiDataRead, NiiDataWrite

if __name__ == '__main__':
    taget_size = (32, 160, 192)
    values_clip = (-55, 145)
    data_root = r'./RAW_DATA/'
    save_root = r'./Processed_DATA/'

    for dataset in ['Train', 'Val']:
        for ID in os.listdir(join(data_root, dataset)):
            print(ID)
            for modal in ['NC', 'AP', 'PVP', 'DP']:
                img, spacing, origin, direction = NiiDataRead(os.path.join(data_root, dataset, ID, f"{modal}.nii.gz"))
                mask_liver, _, _, _ = NiiDataRead(os.path.join(data_root, dataset, ID, 'Liver_mask.nii.gz'))
                mask_tumor, _, _, _ = NiiDataRead(os.path.join(data_root, dataset, ID, 'Tumor_mask.nii.gz'))

                img = np.clip(img, values_clip[0], values_clip[1])
                img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

                img_liver = np.copy(img)
                img_tumor = np.copy(img)
                img_liver[mask_liver == 0] = -1
                img_tumor[mask_tumor == 0] = -1
                os.makedirs(join(save_root, dataset, ID), exist_ok=True)
                NiiDataWrite(join(save_root, dataset, ID, f'{modal}_liver.nii.gz'), img_liver, spacing, origin, direction)
                NiiDataWrite(join(save_root, dataset, ID, f'{modal}_tumor.nii.gz'), img_tumor, spacing, origin, direction)