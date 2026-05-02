import os
import torch
import argparse
import numpy as np
import torch.optim as optim
from datetime import datetime
from collections import Counter
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score

from loss_function.CB_Loss import CB_loss
from util.Nii_utils import setup_seed, Save_Parameter
from data.dataset_3D import Dataset_for_metastatic_tumor
from Networks.ResNet_3D import ResNet18_3D_4stream_clinical_LSTM


def main(args):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.checkpoints_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(os.path.join(args.checkpoints_dir, 'log/val'), flush_secs=2)
    test_writer = SummaryWriter(os.path.join(args.checkpoints_dir, 'log/test'), flush_secs=2)
    wtest_writer = SummaryWriter(os.path.join(args.checkpoints_dir, 'log/wtest'), flush_secs=2)
    print(args.checkpoints_dir)
    Save_Parameter(args)
    print('dataset loading')

    train_data = Dataset_for_metastatic_tumor(args.data_dir, args.train_split_path, args.metadata_path, augment=True)
    val_data = Dataset_for_metastatic_tumor(args.data_dir, args.val_split_path, args.metadata_path, augment=False)
    test_data = Dataset_for_metastatic_tumor(args.data_dir, args.test_split_path, args.metadata_path, augment=False)
    wtest_data = Dataset_for_metastatic_tumor(args.data_dir, args.wtest_split_path, args.metadata_path, augment=False)

    if args.resample_data:
        train_targets = train_data.label_list
        counter = dict(Counter(train_targets))
        class_sample_counts = [counter[i] for i in range(len(counter))]  ####（0类和1类）
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = weights[train_targets]
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, num_workers=min(8, args.bs), shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs*4, num_workers=min(8, args.bs*4), shuffle=False, drop_last=False)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.bs*4, num_workers=min(8, args.bs*4), shuffle=False, drop_last=False)
    wtest_dataloader = DataLoader(dataset=wtest_data, batch_size=args.bs*4, num_workers=min(8, args.bs*4), shuffle=False, drop_last=False)

    print('train_lenth: %i  val_lenth: %i test_lenth: %i wtest_lenth: %i num_0: %i  num_1: %i  num_2: %i  num_3: %i  num_4: %i  num_other: %i' % (
        train_data.len, val_data.len, test_data.len, wtest_data.len, train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3, train_data.num_4, train_data.num_5))

    net = ResNet18_3D_4stream_clinical_LSTM(in_channels=2, clinical_inchannels=3, n_classes=args.num_class, pretrained=True, no_cuda=False).cuda()
    if args.pretrain:
        net.load_state_dict(torch.load("/home/zky/ALMSS-main/trained_models/metastatic_tumor_3D_mask/bs16_epoch100_seed42_May21_16-51-03/best_AUC_val.pth"))
    optimizer = optim.AdamW(net.parameters(), lr=args.lr_max, weight_decay=args.L2)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * args.epoch), int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
    # lr_scheduler = MultiStepLR(optimizer, milestones=[int((1 / 3) * args.epoch), int((2 / 3) * args.epoch)], gamma=0.1, last_epoch=-1)

    best_AUC_val = 0
    best_ACC_val = 0

    best_AUC_test = 0
    best_ACC_test = 0

    best_AUC_wtest = 0
    best_ACC_wtest = 0

    print('training')

    for epoch in range(args.epoch):
        net.train()
        train_epoch_loss = []
        train_epoch_one_hot_label = []
        train_epoch_pred_scores = []
        train_epoch_class_label = []
        train_epoch_pred_class = []
        for i, DATA in enumerate(train_dataloader):
            Plain_imgs = DATA["Plain_img"].cuda().float()
            Arterial_imgs = DATA["Arterial_img"].cuda().float()
            Venous_imgs = DATA["Venous_img"].cuda().float()
            Delay_imgs = DATA["Delay_img"].cuda().float()
            gender_ages, labels = DATA["gender_age"].cuda().float(), DATA["label"].cuda().long()
            labels_one_hot = torch.zeros((labels.size(0), args.num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
            optimizer.zero_grad()
            outputs = net(Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages)
            loss = CB_loss(labels, outputs, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3, train_data.num_4, train_data.num_5],
                           no_of_classes=args.num_class, loss_type='focal', beta=0.999, gamma=2)
            # loss.backward()
            flood_loss = (loss - args.flood).abs() + args.flood
            flood_loss.backward()
            optimizer.step()
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
            train_epoch_pred_scores.append(outputs.detach().cpu())
            train_epoch_one_hot_label.append(labels_one_hot)
            train_epoch_loss.append(loss.item())
            train_epoch_class_label.append(labels.cpu().numpy())
            train_epoch_pred_class.append(predicted.cpu().numpy())
            print('[%d/%d, %d/%d] train_loss: %.3f' %
                  (epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
        lr_scheduler.step()

        with torch.no_grad():
            net.eval()
            val_epoch_loss = []
            val_epoch_label = []
            val_epoch_pred_scores = []
            val_epoch_class_label = []
            val_epoch_pred_class = []
            for i, DATA in enumerate(val_dataloader):
                Plain_imgs = DATA["Plain_img"].cuda().float()
                Arterial_imgs = DATA["Arterial_img"].cuda().float()
                Venous_imgs = DATA["Venous_img"].cuda().float()
                Delay_imgs = DATA["Delay_img"].cuda().float()
                gender_ages, labels = DATA["gender_age"].cuda().float(), DATA["label"].cuda().long()
                labels_one_hot = torch.zeros((labels.size(0), args.num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
                outputs = net(Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages)
                loss = CB_loss(labels, outputs,
                               samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3,
                                                train_data.num_4, train_data.num_5],
                               no_of_classes=args.num_class, loss_type='focal', beta=0.999, gamma=2)
                outputs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
                val_epoch_pred_scores.append(outputs.detach().cpu())
                val_epoch_label.append(labels_one_hot)
                val_epoch_loss.append(loss.item())
                val_epoch_class_label.append(labels.cpu().numpy())
                val_epoch_pred_class.append(predicted.cpu().numpy())

        with torch.no_grad():
            net.eval()
            test_epoch_loss = []
            test_epoch_label = []
            test_epoch_pred_scores = []
            test_epoch_class_label = []
            test_epoch_pred_class = []
            for i, DATA in enumerate(test_dataloader):
                Plain_imgs = DATA["Plain_img"].cuda().float()
                Arterial_imgs = DATA["Arterial_img"].cuda().float()
                Venous_imgs = DATA["Venous_img"].cuda().float()
                Delay_imgs = DATA["Delay_img"].cuda().float()
                gender_ages, labels = DATA["gender_age"].cuda().float(), DATA["label"].cuda().long()
                labels_one_hot = torch.zeros((labels.size(0), args.num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
                outputs = net(Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages)
                loss = CB_loss(labels, outputs,
                               samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3,
                                                train_data.num_4, train_data.num_5],
                               no_of_classes=args.num_class, loss_type='focal', beta=0.999, gamma=2)
                outputs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
                test_epoch_pred_scores.append(outputs.detach().cpu())
                test_epoch_label.append(labels_one_hot)
                test_epoch_loss.append(loss.item())
                test_epoch_class_label.append(labels.cpu().numpy())
                test_epoch_pred_class.append(predicted.cpu().numpy())

        with torch.no_grad():
            net.eval()
            wtest_epoch_loss = []
            wtest_epoch_label = []
            wtest_epoch_pred_scores = []
            wtest_epoch_class_label = []
            wtest_epoch_pred_class = []
            for i, DATA in enumerate(wtest_dataloader):
                Plain_imgs = DATA["Plain_img"].cuda().float()
                Arterial_imgs = DATA["Arterial_img"].cuda().float()
                Venous_imgs = DATA["Venous_img"].cuda().float()
                Delay_imgs = DATA["Delay_img"].cuda().float()
                gender_ages, labels = DATA["gender_age"].cuda().float(), DATA["label"].cuda().long()
                labels_one_hot = torch.zeros((labels.size(0), args.num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
                outputs = net(Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages)
                loss = CB_loss(labels, outputs,
                               samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3,
                                                train_data.num_4, train_data.num_5],
                               no_of_classes=args.num_class, loss_type='focal', beta=0.999, gamma=2)
                outputs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
                wtest_epoch_pred_scores.append(outputs.detach().cpu())
                wtest_epoch_label.append(labels_one_hot)
                wtest_epoch_loss.append(loss.item())
                wtest_epoch_class_label.append(labels.cpu().numpy())
                wtest_epoch_pred_class.append(predicted.cpu().numpy())

        train_epoch_one_hot_label = torch.cat(train_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
        train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()
        val_epoch_label = torch.cat(val_epoch_label, dim=0).numpy().astype(np.uint8)
        val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()
        test_epoch_label = torch.cat(test_epoch_label, dim=0).numpy().astype(np.uint8)
        test_epoch_pred_scores = torch.cat(test_epoch_pred_scores, dim=0).numpy()
        wtest_epoch_label = torch.cat(wtest_epoch_label, dim=0).numpy().astype(np.uint8)
        wtest_epoch_pred_scores = torch.cat(wtest_epoch_pred_scores, dim=0).numpy()

        train_epoch_class_label = np.concatenate(train_epoch_class_label)
        train_epoch_pred_class = np.concatenate(train_epoch_pred_class)
        val_epoch_class_label = np.concatenate(val_epoch_class_label)
        val_epoch_pred_class = np.concatenate(val_epoch_pred_class)
        test_epoch_class_label = np.concatenate(test_epoch_class_label)
        test_epoch_pred_class = np.concatenate(test_epoch_pred_class)
        wtest_epoch_class_label = np.concatenate(wtest_epoch_class_label)
        wtest_epoch_pred_class = np.concatenate(wtest_epoch_pred_class)

        train_AUC = roc_auc_score(train_epoch_one_hot_label, train_epoch_pred_scores)
        val_AUC = roc_auc_score(val_epoch_label, val_epoch_pred_scores)
        test_AUC = roc_auc_score(test_epoch_label, test_epoch_pred_scores)
        try:
            wtest_AUC = roc_auc_score(wtest_epoch_label, wtest_epoch_pred_scores)
        except:
            wtest_AUC = roc_auc_score(np.concatenate((wtest_epoch_label[:, :2], wtest_epoch_label[:, 3:]), axis=1),
                                np.concatenate((wtest_epoch_pred_scores[:, :2], wtest_epoch_pred_scores[:, 3:]), axis=1),
                                multi_class='ovr')

        train_ACC = accuracy_score(train_epoch_class_label, train_epoch_pred_class)
        val_ACC = accuracy_score(val_epoch_class_label, val_epoch_pred_class)
        test_ACC = accuracy_score(test_epoch_class_label, test_epoch_pred_class)
        wtest_ACC = accuracy_score(wtest_epoch_class_label, wtest_epoch_pred_class)

        train_epoch_loss = np.mean(train_epoch_loss)
        val_epoch_loss = np.mean(val_epoch_loss)
        test_epoch_loss = np.mean(test_epoch_loss)
        wtest_epoch_loss = np.mean(wtest_epoch_loss)

        print(
            '[%d/%d] train_loss: %.3f train_AUC: %.3f val_AUC: %.3f test_AUC: %.3f wtest_AUC:%.3f train_ACC: %.3f val_ACC: %.3f test_ACC: %.3f wtest_ACC: %.3f' %
            (epoch, args.epoch, train_epoch_loss, train_AUC, val_AUC, test_AUC, wtest_AUC, train_ACC, val_ACC, test_ACC, wtest_ACC))

        if val_AUC > best_AUC_val:
            best_AUC_val = val_AUC
            torch.save(net.state_dict(), os.path.join(args.checkpoints_dir, 'best_AUC_val.pth'))
        if val_ACC > best_ACC_val:
            best_ACC_val = val_ACC
            torch.save(net.state_dict(), os.path.join(args.checkpoints_dir, 'best_ACC_val.pth'))
        if test_AUC > best_AUC_test:
            best_AUC_test = test_AUC
            torch.save(net.state_dict(), os.path.join(args.checkpoints_dir, 'best_AUC_test.pth'))
        if test_ACC > best_ACC_test:
            best_ACC_test = test_ACC
            torch.save(net.state_dict(), os.path.join(args.checkpoints_dir, 'best_ACC_test.pth'))
        if wtest_AUC > best_AUC_wtest:
            best_AUC_wtest = wtest_AUC
        if wtest_ACC > best_ACC_wtest:
            best_ACC_wtest = wtest_ACC
        if epoch + 1 == args.epoch:
            torch.save(net.state_dict(), os.path.join(args.checkpoints_dir, 'epoch' + str(epoch + 1) + '.pth'))

        train_writer.add_scalar('loss', train_epoch_loss, epoch)
        train_writer.add_scalar('AUC', train_AUC, epoch)
        train_writer.add_scalar('ACC', train_ACC, epoch)

        val_writer.add_scalar('loss', val_epoch_loss, epoch)
        val_writer.add_scalar('AUC', val_AUC, epoch)
        val_writer.add_scalar('ACC', val_ACC, epoch)
        val_writer.add_scalar('best_AUC_val', best_AUC_val, epoch)
        val_writer.add_scalar('best_ACC_val', best_ACC_val, epoch)

        test_writer.add_scalar('loss', test_epoch_loss, epoch)
        test_writer.add_scalar('AUC', test_AUC, epoch)
        test_writer.add_scalar('ACC', test_ACC, epoch)
        test_writer.add_scalar('best_AUC_test', best_AUC_test, epoch)
        test_writer.add_scalar('best_ACC_test', best_ACC_test, epoch)

        wtest_writer.add_scalar('loss', wtest_epoch_loss, epoch)
        wtest_writer.add_scalar('AUC', wtest_AUC, epoch)
        wtest_writer.add_scalar('ACC', wtest_ACC, epoch)
        wtest_writer.add_scalar('best_AUC_wtest', best_AUC_wtest, epoch)
        wtest_writer.add_scalar('best_ACC_wtest', best_ACC_wtest, epoch)


    print('saved_model_name:', args.checkpoints_dir)
    try:
        os.rename(args.checkpoints_dir, args.checkpoints_dir + '_val' + str(best_AUC_val) + '_test' + str(best_AUC_test) + '_wtest' + str(best_AUC_wtest))
    except:
        print('rename error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--bs', type=int, default=24, help='batch size')
    # parser.add_argument('--bs', type=int, default=16, help='batch size')
    # parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--epoch', type=int, default=60, help='all_epochs')
    parser.add_argument('--seed', type=int, default=56, help='random seed')
    parser.add_argument('--flood', type=float, default=0.2, help='random seed')
    parser.add_argument('--lr_max', type=float, default=0.0002, help='random seed')
    parser.add_argument('--data_dir', type=str, default='data/Liver_CE_classifiy_data_preprocessed/')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--resample_data', type=bool, default=False)
    parser.add_argument('--num_class', type=int, default=6)
    args = parser.parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.checkpoints_dir = 'trained_models/metastatic_tumor_3D/bs{}_epoch{}_seed{}_{}'.format(args.bs, args.epoch, args.seed, current_time)
    args.L2 = 0.00005
    args.input_size = (32, 160, 192)
    args.metadata_path = '../relevant_files/metadata.xlsx'
    args.train_split_path = '../relevant_files/train.txt'
    args.val_split_path = '../relevant_files/val.txt'
    args.test_split_path = '../relevant_files/test.txt'
    args.wtest_split_path = '../relevant_files/wtest.txt'
    main(args)