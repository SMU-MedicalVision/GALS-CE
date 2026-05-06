import os
import torch
import numpy as np
import pandas as pd
from os.path import join
import torch.utils.data as data
from Nii_utils import NiiDataRead
from volumentations import *


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, metadata_path, augment=False, dataset='Train', label_type='primary_metastatic'):
        self.data_dir = data_dir
        self.dataset = dataset
        self.augment = augment

        ID_list_orginal = os.listdir(join(self.data_dir, dataset))

        metadata_df = pd.read_excel(metadata_path)
        metadata_df['ID'] = metadata_df['ID'].astype(str)
        metadata_df['label'] = metadata_df['label'].astype(int)

        self.ID_list = []
        self.gender_list = []
        self.age_list = []
        self.label_list = []

        for ID in ID_list_orginal:
            label_one = metadata_df.loc[metadata_df.ID == ID, 'label'].values[0]

            if label_type=='primary':
                label = 0 if label_one <= 6 else label_one - 6
            elif label_type=='metastatic':
                if label_one <= 6:
                    label = label_one-1
                else:
                    continue
            elif label_type=='primary_metastatic':
                label = label_one-1

            self.ID_list.append(ID)
            self.label_list.append(label)
            self.gender_list.append(str(metadata_df.loc[metadata_df.ID == ID, 'sex'].values[0]))
            self.age_list.append(int(metadata_df.loc[metadata_df.ID == ID, 'age'].values[0]))


        self.num_0 = self.label_list.count(0)
        self.num_1 = self.label_list.count(1)
        self.num_2 = self.label_list.count(2)
        if label_type in ['metastatic', 'primary_metastatic']:
            self.num_3 = self.label_list.count(3)
            self.num_4 = self.label_list.count(4)
            self.num_5 = self.label_list.count(5)
        if label_type in ['primary_metastatic']:
            self.num_6 = self.label_list.count(6)
            self.num_7 = self.label_list.count(7)


        self.transforms = Compose([
            RotatePseudo2D(axes=(1, 2), limit=(-30, 30), interpolation=3, value=-1, p=0.3),
            ElasticTransformPseudo2D(alpha=50, sigma=30, alpha_affine=10, value=-1, p=0.3),
            GaussianNoise(var_limit=(0, 0.1), mean=0, p=0.3),
        ])

        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        gender = self.gender_list[idx]
        age = self.age_list[idx]
        label = self.label_list[idx]

        if gender == 'male':
            gender = torch.from_numpy(np.array([1, 0]))
        elif gender == 'female':
            gender = torch.from_numpy(np.array([0, 1]))

        age = torch.from_numpy(np.array([age])).float() / 100

        NC_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'NC_liver.nii.gz'))
        AP_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'AP_liver.nii.gz'))
        PVP_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'PVP_liver.nii.gz'))
        DP_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'DP_liver.nii.gz'))

        NC_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'NC_tumor.nii.gz'))
        AP_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'AP_tumor.nii.gz'))
        PVP_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'PVP_tumor.nii.gz'))
        DP_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, self.dataset, ID, 'DP_tumor.nii.gz'))
        img = np.concatenate((NC_liver[..., np.newaxis], NC_tumor[..., np.newaxis],
                              AP_liver[..., np.newaxis], AP_tumor[..., np.newaxis],
                              PVP_liver[..., np.newaxis], PVP_tumor[..., np.newaxis],
                              DP_liver[..., np.newaxis], DP_tumor[..., np.newaxis]), axis=-1)

        if self.augment:
            img = self.transforms(image=img)['image']
            age += (torch.rand(*age.size()) - 0.5) * 0.04

        img = torch.from_numpy(img).permute(3, 0, 1, 2)

        NC = img[0:2]
        AP = img[2:4]
        PVP = img[4:6]
        DP = img[6:8]
        gender_age = torch.cat((gender, age))
        label = torch.tensor(label)
        return {"NC": NC,
                "AP": AP,
                "PVP": PVP,
                "DP": DP,
                "gender_age": gender_age,
                "label": label,
                }

    def __len__(self):
        return self.len





