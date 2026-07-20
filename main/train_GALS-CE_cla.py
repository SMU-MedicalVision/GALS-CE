import warnings
warnings.filterwarnings("ignore")
import os
import torch
import pathlib
import argparse
import numpy as np
import pandas as pd
from os.path import join
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*To copy construct from a tensor.*')
from Nii_utils import setup_seed, Save_Parameter
from Networks.Networks_cla.ResNet_3D import ResNet18_3D_4stream_clinical_LSTM, ResNet18_3D_4stream_clinical_LSTM_latefusion
from loss_function.CB_Loss import CB_loss
from data.dataset_cla import DatasetFromFolder
from sklearn.metrics import roc_auc_score, accuracy_score


import os
import numpy as np
from os.path import join
import SimpleITK as sitk
from skimage import transform


def NiiDataRead(path, image_only=False, as_type=np.float32):
    """
    Read a NIfTI medical image and return its data, spacing, origin, and direction.

    Args:
        path (str): Path to the NIfTI file.
        as_type (np.dtype): Desired data type of the returned image array.

    Returns:
        volumn (ndarray): 3D image data in [z, y, x] order.
        spacing_ (ndarray): Voxel spacing in [z, y, x] order.
        origin (tuple): Image origin in physical space.
        direction (tuple): Image orientation (direction cosine matrix).
    """
    nii = sitk.ReadImage(path)
    spacing = nii.GetSpacing()  # [x,y,z]
    volumn = sitk.GetArrayFromImage(nii)  # [z,y,x]
    if image_only:
        return volumn.astype(as_type)
    else:
        origin = nii.GetOrigin()
        direction = nii.GetDirection()

        spacing_x = spacing[0]
        spacing_y = spacing[1]
        spacing_z = spacing[2]

        spacing_ = np.array([spacing_z, spacing_y, spacing_x])
        return volumn.astype(as_type), spacing_.astype(np.float32), origin, direction


def NiiDataWrite(save_path, volumn, spacing=np.array([1,1,1]), origin=(0, 0, 0), direction=[1, 0, 0, 0, 1, 0, 0, 0, 1], as_type=np.float32):
    """
    Save a 3D numpy array as a NIfTI file with spatial information.

    Args:
        save_path (str): Destination path to save the NIfTI file.
        volumn (ndarray): 3D image data [z, y, x].
        spacing (ndarray): Spacing in [z, y, x] order.
        origin (tuple): Image origin.
        direction (tuple): Image orientation.
        as_type (np.dtype): Data type to save as.
    """
    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn[:, :, :].astype(as_type))

    # Convert spacing back to [x, y, z] for SimpleITK
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    raw.SetOrigin(origin)
    raw.SetDirection(direction)
    sitk.WriteImage(raw, save_path)


def Preprocess(phase_data_save_dict):
    taget_size = (32, 160, 192)
    values_clip = (-55, 145)

    for modal in ['NC', 'AP', 'PVP', 'DP']:
        img, spacing, origin, direction = NiiDataRead(join(phase_data_save_dict[modal][0], f"{modal}.nii.gz"))

        mask_liver = NiiDataRead(join(phase_data_save_dict['mask'], 'Liver_mask.nii.gz'), image_only=True)
        mask_tumor = NiiDataRead(join(phase_data_save_dict['mask'], 'Tumor_mask.nii.gz'), image_only=True)

        z_, x_, y_ = mask_liver.nonzero()
        z1 = z_.min()
        z2 = z_.max()
        x1 = x_.min()
        x2 = x_.max()
        y1 = y_.min()
        y2 = y_.max()

        img = img[z1: z2 + 1, x1: x2 + 1, y1: y2 + 1]
        mask_liver = mask_liver[z1: z2 + 1, x1: x2 + 1, y1: y2 + 1]
        mask_tumor = mask_tumor[z1: z2 + 1, x1: x2 + 1, y1: y2 + 1]

        img = np.clip(img, values_clip[0], values_clip[1])
        img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

        # 重新计算 spacing
        spacing_z = (spacing[0] * img.shape[0]) / taget_size[0]
        spacing_x = (spacing[1] * img.shape[1]) / taget_size[1]
        spacing_y = (spacing[2] * img.shape[2]) / taget_size[2]
        spacing = np.array([spacing_z, spacing_x, spacing_y])


        img = transform.resize(img, taget_size, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)
        mask_liver = transform.resize(mask_liver, taget_size, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)
        mask_tumor = transform.resize(mask_tumor, taget_size, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)

        img_liver = np.copy(img)
        img_tumor = np.copy(img)
        img_liver[mask_liver == 0] = -1
        img_tumor[mask_tumor == 0] = -1
        os.makedirs(join(phase_data_save_dict[modal][1]), exist_ok=True)
        NiiDataWrite(join(phase_data_save_dict[modal][1], f'{modal}_liver.nii.gz'), img_liver, spacing, origin, direction)
        NiiDataWrite(join(phase_data_save_dict[modal][1], f'{modal}_tumor.nii.gz'), img_tumor, spacing, origin, direction)
    print(join(phase_data_save_dict[modal][1], f'{modal}_liver.nii.gz'))
    print(join(phase_data_save_dict[modal][1], f'{modal}_tumor.nii.gz'))


def main(opt, pretrain=False, label_type='primary_metastatic'):
    pathlib.Path(opt.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(join(opt.checkpoints_dir, 'log/train')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(join(opt.checkpoints_dir, 'log/val')).mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(join(opt.checkpoints_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(join(opt.checkpoints_dir, 'log/val'), flush_secs=2)

    Save_Parameter(opt)

    train_data = DatasetFromFolder(opt.data_dir, opt.metadata_path, augment=False, dataset='Train', label_type=label_type)
    val_data = DatasetFromFolder(opt.data_dir, opt.metadata_path, augment=False, dataset='Val', label_type=label_type)
    num_train_list = [getattr(train_data, f'num_{i}', 0) for i in range(opt.num_class)]
    num_val_list = [getattr(val_data, f'num_{i}', 0) for i in range(opt.num_class)]

    train_dataloader = DataLoader(dataset=train_data, batch_size=opt.bs, num_workers=opt.num_threads, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(dataset=val_data, batch_size=opt.val_bs, num_workers=opt.num_threads, shuffle=False, drop_last=False)

    print(f'train_lenth: {train_data.len} |' + '  '.join([f'num_{i}: {num_train_list[i]}' for i in range(opt.num_class)]))
    print(f'val_lenth: {val_data.len} |' + '  '.join([f'num_{i}: {num_val_list[i]}' for i in range(opt.num_class)]))

    if pretrain:
        net = ResNet18_3D_4stream_clinical_LSTM(in_channels=2, clinical_inchannels=3, n_classes=opt.num_class, pretrained=False, no_cuda=False).cuda()
    else:
        net = ResNet18_3D_4stream_clinical_LSTM_latefusion(in_channels=2, clinical_inchannels=3, n_classes=[3, 6], pretrain_pth={'primary':opt.primary_model_pth, 'metastatic':opt.metastatic_model_pth}, no_cuda=False).cuda()
        for name, param in net.named_parameters():
            if 'classifer' in name:
                param.requires_grad = False

    optimizer = optim.AdamW(net.parameters(), lr=opt.lr_max, weight_decay=opt.L2)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * opt.max_epoch), int((9 / 10) * opt.max_epoch)], gamma=0.1, last_epoch=-1)

    best_AUC_val = 0
    best_ACC_val = 0
    best_val_epoch = 0


    for epoch in range(opt.max_epoch):
        net.train()
        train_epoch_loss = []
        train_epoch_one_hot_label = []
        train_epoch_pred_scores = []
        train_epoch_class_label = []
        train_epoch_pred_class = []
        for i, DATA in enumerate(train_dataloader):
            NC = DATA['NC'].cuda().float()
            AP = DATA['AP'].cuda().float()
            PVP = DATA['PVP'].cuda().float()
            DP = DATA['DP'].cuda().float()
            gender_ages, labels = DATA['gender_age'].cuda().float(), DATA['label'].cuda().long()
            labels_one_hot = torch.zeros((labels.size(0), opt.num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
            optimizer.zero_grad()
            outputs = net(NC, AP, PVP, DP, gender_ages)
            loss = CB_loss(labels, outputs, samples_per_cls=num_train_list, no_of_classes=opt.num_class, loss_type='focal', beta=0.999, gamma=2)
            if opt.flood>0:
                flood_loss = (loss - opt.flood).abs() + opt.flood
                flood_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            outputs_softmax = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs_softmax, dim=1, keepdim=False).detach()
            train_epoch_pred_scores.append(outputs_softmax.detach().cpu())
            train_epoch_one_hot_label.append(labels_one_hot)
            train_epoch_loss.append(loss.item())
            train_epoch_class_label.append(labels.cpu().numpy())
            train_epoch_pred_class.append(predicted.cpu().numpy())
        lr_scheduler.step()

        with torch.no_grad():
            net.eval()
            val_epoch_loss = []
            val_epoch_label = []
            val_epoch_pred_scores = []
            val_epoch_class_label = []
            val_epoch_pred_class = []
            for i, DATA in enumerate(val_dataloader):
                NC = DATA['NC'].cuda().float()
                AP = DATA['AP'].cuda().float()
                PVP = DATA['PVP'].cuda().float()
                DP = DATA['DP'].cuda().float()
                gender_ages, labels = DATA['gender_age'].cuda().float(), DATA['label'].cuda().long()
                labels_one_hot = torch.zeros((labels.size(0), opt.num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
                outputs = net(NC, AP, PVP, DP, gender_ages)
                loss = CB_loss(labels, outputs, samples_per_cls=[1]*opt.num_class, no_of_classes=opt.num_class, loss_type='focal', beta=0.999, gamma=2)
                outputs_softmax = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs_softmax, dim=1, keepdim=False).detach()
                val_epoch_pred_scores.append(outputs_softmax.detach().cpu())
                val_epoch_label.append(labels_one_hot)
                val_epoch_loss.append(loss.item())
                val_epoch_class_label.append(labels.cpu().numpy())
                val_epoch_pred_class.append(predicted.cpu().numpy())


        train_epoch_one_hot_label = torch.cat(train_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
        train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()
        val_epoch_label = torch.cat(val_epoch_label, dim=0).numpy().astype(np.uint8)
        val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()

        train_epoch_class_label = np.concatenate(train_epoch_class_label)
        train_epoch_pred_class = np.concatenate(train_epoch_pred_class)
        val_epoch_class_label = np.concatenate(val_epoch_class_label)
        val_epoch_pred_class = np.concatenate(val_epoch_pred_class)

        try:
            train_AUC = roc_auc_score(train_epoch_one_hot_label, train_epoch_pred_scores)
        except:
            train_AUC = 0
        try:
            val_AUC = roc_auc_score(val_epoch_label, val_epoch_pred_scores)
        except:
            val_AUC = 0

        train_ACC = accuracy_score(train_epoch_class_label, train_epoch_pred_class)
        val_ACC = accuracy_score(val_epoch_class_label, val_epoch_pred_class)

        train_epoch_loss = np.mean(train_epoch_loss)
        val_epoch_loss = np.mean(val_epoch_loss)

        if val_AUC > best_AUC_val:
            try:
                os.remove(join(opt.checkpoints_dir, f'best_AUC_val_{best_val_epoch}.pth'))
            except:
                pass
            torch.save(net.state_dict(), join(opt.checkpoints_dir, f'best_AUC_val_{best_val_epoch}.pth'))
            best_AUC_val = val_AUC
            best_val_epoch = epoch
        if val_ACC > best_ACC_val:
            best_ACC_val = val_ACC

        try:
            os.remove(join(opt.checkpoints_dir, 'epoch' + str(epoch-1) + '.pth'))
        except:
            pass
        torch.save(net.state_dict(), join(opt.checkpoints_dir, 'epoch' + str(epoch) + '.pth'))

        train_writer.add_scalar('loss', train_epoch_loss, epoch)
        train_writer.add_scalar('AUC', train_AUC, epoch)
        train_writer.add_scalar('ACC', train_ACC, epoch)

        val_writer.add_scalar('loss', val_epoch_loss, epoch)
        val_writer.add_scalar('AUC', val_AUC, epoch)
        val_writer.add_scalar('ACC', val_ACC, epoch)
        val_writer.add_scalar('best_AUC_val', best_AUC_val, epoch)
        val_writer.add_scalar('best_ACC_val', best_ACC_val, epoch)

    train_writer.flush()
    val_writer.flush()
    train_writer.close()
    val_writer.close()
    try:
        os.rename(opt.checkpoints_dir, opt.checkpoints_dir + '_val' + str(best_AUC_val))
        opt.checkpoints_dir = opt.checkpoints_dir + '_val' + str(best_AUC_val)
    except:
        print('rename error')
    best_model_pth = join(opt.checkpoints_dir, f'best_AUC_val_{best_val_epoch}.pth')
    if not os.path.exists(best_model_pth):
        best_model_pth = join(opt.checkpoints_dir, 'epoch' + str(epoch) + '.pth')
    assert os.path.exists(best_model_pth), 'Best model path does not exist: {}'.format(best_model_pth)
    return net, best_model_pth

def pred(opt, net):
    net.load_state_dict(torch.load(opt.train_model_pth), strict=True)
    net.eval()
    metadata_path = f'./RAW_DATA/metadata_{opt.inf_dataset}.xlsx'
    metadata_df = pd.read_excel(metadata_path)
    ID_list_orginal = os.listdir(join(opt.raw_dir, f'{opt.inf_dataset}_GALS-CE_syn'))
    print('testing')
    with torch.no_grad():
        net.eval()
        ID_list = []
        gender_list = []
        age_list = []
        for ID in ID_list_orginal:
            ID_list.append(ID)
            gender_list.append(str(metadata_df.loc[metadata_df.ID == ID, 'sex'].values[0]))
            age_list.append(int(metadata_df.loc[metadata_df.ID == ID, 'age'].values[0]))
        pred_scores_all = []
        pred_class_all = []

        for i, ID in enumerate(ID_list):
            gender = gender_list[i]
            age = age_list[i]

            if gender == 'male':
                gender = torch.from_numpy(np.array([1, 0]))
            elif gender == 'female':
                gender = torch.from_numpy(np.array([0, 1]))
            else:
                print('Wrong!!')
            age = torch.from_numpy(np.array([age])).float() / 100

            NC_processed_path = join(opt.data_dir, opt.inf_dataset, ID)
            syn_processed_path = join(opt.data_dir, f'{opt.inf_dataset}_GALS-CE_syn', ID)
            if (not os.path.exists(NC_processed_path)) or (not os.path.exists(syn_processed_path)):
                phase_data_save_dict = {'NC': [join(opt.raw_dir, opt.inf_dataset, ID), NC_processed_path],
                                        'mask':join(opt.raw_dir, opt.inf_dataset, ID),
                                        'AP':[join(opt.raw_dir, f'{opt.inf_dataset}_GALS-CE_syn', ID), syn_processed_path],
                                        'PVP':[join(opt.raw_dir, f'{opt.inf_dataset}_GALS-CE_syn', ID), syn_processed_path],
                                        'DP':[join(opt.raw_dir, f'{opt.inf_dataset}_GALS-CE_syn', ID), syn_processed_path]}
                Preprocess(phase_data_save_dict)
                assert os.path.exists(NC_processed_path) and os.path.exists(syn_processed_path), 'Preprocess error, check the preprocess function'
            else:
                print('procress skip')


            NC_liver = NiiDataRead(join(NC_processed_path, 'NC_liver.nii.gz'), image_only=True)
            NC_tumor = NiiDataRead(join(NC_processed_path, 'NC_tumor.nii.gz'), image_only=True)
            AP_liver = NiiDataRead(join(syn_processed_path, 'AP_liver.nii.gz'), image_only=True)
            AP_tumor = NiiDataRead(join(syn_processed_path, 'AP_tumor.nii.gz'), image_only=True)
            PVP_liver = NiiDataRead(join(syn_processed_path, 'PVP_liver.nii.gz'), image_only=True)
            PVP_tumor = NiiDataRead(join(syn_processed_path, 'PVP_tumor.nii.gz'), image_only=True)
            DP_liver = NiiDataRead(join(syn_processed_path, 'DP_liver.nii.gz'), image_only=True)
            DP_tumor = NiiDataRead(join(syn_processed_path, 'DP_tumor.nii.gz'), image_only=True)

            NC = torch.from_numpy(
                np.concatenate((NC_liver[np.newaxis, ...], NC_tumor[np.newaxis, ...]), axis=0)).unsqueeze(0).float().cuda()
            AP = torch.from_numpy(
                np.concatenate((AP_liver[np.newaxis, ...], AP_tumor[np.newaxis, ...]),axis=0)).unsqueeze(0).float().cuda()
            PVP = torch.from_numpy(
                np.concatenate((PVP_liver[np.newaxis, ...], PVP_tumor[np.newaxis, ...]),axis=0)).unsqueeze(0).float().cuda()
            DP = torch.from_numpy(
                np.concatenate((DP_liver[np.newaxis, ...], DP_tumor[np.newaxis, ...]), axis=0)).unsqueeze(0).float().cuda()

            gender_age = torch.cat((gender, age)).unsqueeze(0).float().cuda()

            output = net(NC, AP, PVP, DP, gender_age)
            output = torch.softmax(output, dim=1)

            predicted = torch.argmax(output, dim=1, keepdim=False).detach()
            pred_scores_all.append(output.detach().cpu())
            pred_class_all.append(predicted.cpu().numpy())

        pred_scores_all = torch.cat(pred_scores_all, dim=0).numpy()
        pred_class_all = np.concatenate(pred_class_all)

        df = pd.DataFrame({
            'name': ID_list,
            'pred': pred_class_all,
            **{f'pred_score{i}': pred_scores_all[:, i] for i in range(opt.num_class)}
        })

        df.to_excel(join(opt.data_dir, f'GALS-CE_cla_pred_{opt.inf_dataset}.xlsx'), index=False)
    print('Inference results save to:', join(opt.data_dir, f'GALS-CE_cla_pred_{opt.inf_dataset}.xlsx'))


if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='1', help='which gpu is used')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--num_threads', type=int, default=8, help='# threads for loading data')
    parser.add_argument('--max_epoch', type=int, default=50, help='all_epochs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--flood', type=float, default=0.2, help='random seed')
    parser.add_argument('--lr_max', type=float, default=0.0002, help='random seed')
    parser.add_argument('--data_dir', type=str, default='./Processed_DATA')
    parser.add_argument('--raw_dir', type=str, default='./RAW_DATA')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--num_class', type=int, default=8)
    # -------------------- Inference settings
    parser.add_argument('--inf_dataset', type=str, default='Inference', help='./main/RAW_DATA/{inf_dataset}')
    parser.add_argument('--val_bs', type=int, default=16, help='Val/Test batch size')
    parser.add_argument('--train_model_pth', type=str, default='', help='trained model path for inference')
    # # -------------------- Quick test settings
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')
    opt = parser.parse_args()
    setup_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if opt.quick_test:
        opt.max_epoch = 1


    opt.L2 = 0.00005
    opt.input_size = (32, 160, 192)
    opt.metadata_path = './RAW_DATA/metadata.xlsx'
    setup_seed(opt.seed)

    # -------------- Experiment naming & directory setup --------------
    if not opt.inference_only:
        opt.checkpoints_dir = './main/trained_models/GALS-CE_cla/pretrain/primary/bs{}_epoch{}_seed{}_{}'.format(opt.bs, opt.max_epoch, opt.seed, current_time)
        opt.save_dir = opt.checkpoints_dir
        pathlib.Path(opt.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        # print('-' * 50, "\nCBSI_cla (primary) Pretraining Start\n", '-' * 50)
        opt.num_class=3
        print('-' * 50, "\nCBSI_cla (primary) Pretraining Start\n")
        _, pretrain_model_pth = main(opt, pretrain=True, label_type='primary')
        opt.primary_model_pth = pretrain_model_pth
        print("\nCBSI_cla (primary) Pretraining Done\n", '-' * 50)

        opt.checkpoints_dir = './main/trained_models/GALS-CE_cla/pretrain/metastatic/bs{}_epoch{}_seed{}_{}'.format(opt.bs, opt.max_epoch, opt.seed, current_time)
        opt.save_dir = opt.checkpoints_dir
        pathlib.Path(opt.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        # print('-' * 50, "\nCBSI_cla (metastatic) Pretraining Start\n", '-' * 50)
        opt.num_class=6
        print("\nCBSI_cla (metastatic) Pretraining Start\n")
        _, pretrain_model_pth = main(opt, pretrain=True, label_type='metastatic')
        opt.metastatic_model_pth = pretrain_model_pth
        print("\nCBSI_cla (metastatic) Pretraining Done\n", '-' * 50)

    if not opt.inference_only:
        opt.num_class = 8
        opt.checkpoints_dir = './main/trained_models/GALS-CE_cla/train/pretrain_freeze_primary_metastatic/bs{}_epoch{}_seed{}_{}'.format(opt.bs, opt.max_epoch, opt.seed, current_time)
        opt.save_dir = opt.checkpoints_dir
        pathlib.Path(opt.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        # print('-' * 50, "\nCBSI_cla (primary_metastatic) Training Start\n", '-' * 50)
        net, train_model_pth = main(opt, pretrain=False, label_type='primary_metastatic')
        print('-' * 50, "\nCBSI_cla (primary_metastatic) Training Done\n", '-' * 50)
        opt.train_model_pth = train_model_pth
    else:
        net = ResNet18_3D_4stream_clinical_LSTM_latefusion(in_channels=2, clinical_inchannels=3, n_classes=[3, 6], no_cuda=False).cuda()

    # print('-' * 50, "\nCBSI_cla Inference Start\n", '-' * 50)
    pred(opt, net)
    print('-' * 50, "\nCBSI_cla Inference Done\n", '-' * 50)
