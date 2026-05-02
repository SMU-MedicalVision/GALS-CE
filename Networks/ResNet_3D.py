import torch
from torch import nn
from Networks import resnet3d





class ResNet18_3D_4stream_clinical_LSTM(nn.Module):
    def __init__(self, in_channels, n_classes, clinical_inchannels, pretrained=True, no_cuda=False):
        super(ResNet18_3D_4stream_clinical_LSTM, self).__init__()
        self.backbone1 = resnet3d.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1, shortcut_type='A',
            no_cuda=no_cuda, num_seg_classes=2)
        self.backbone2 = resnet3d.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1, shortcut_type='A',
            no_cuda=no_cuda, num_seg_classes=2)
        self.backbone3 = resnet3d.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1, shortcut_type='A',
            no_cuda=no_cuda, num_seg_classes=2)
        self.backbone4 = resnet3d.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1, shortcut_type='A',
                                       no_cuda=no_cuda, num_seg_classes=2)
        if pretrained:
            net_dict = self.backbone1.state_dict()
            if torch.cuda.is_available():
                pretrain = torch.load(r'Networks/resnet_18_23dataset.pth')
            else:
                pretrain = torch.load(r'Networks/resnet_18_23dataset.pth', map_location='cpu')
            pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict.keys()}
            if in_channels != 1:
                keys_to_remove = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                                  'bn1.num_batches_tracked']
                for key in keys_to_remove:
                    pretrain_dict.pop(key, None)
            net_dict.update(pretrain_dict)
            self.backbone1.load_state_dict(net_dict)
            self.backbone2.load_state_dict(net_dict)
            self.backbone3.load_state_dict(net_dict)
            self.backbone4.load_state_dict(net_dict)

        self.max_pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        self.LSTM = nn.LSTM(input_size=100, hidden_size=512, num_layers=2, bias=True,
                            batch_first=True, dropout=0)

        self.FDR = nn.Linear(512, 100)
        self.gender_age = nn.Sequential(
            nn.Linear(clinical_inchannels, 100, bias=True)
        )
        self.mp = nn.MaxPool1d(4, stride=1)
        self.last_linear = nn.Linear(512, n_classes)

    def feature_extraction(self, x, backbone):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        return x

    def forward(self, x1, x2, x3, x4, ga):
        x1 = self.feature_extraction(x1, self.backbone1)
        x2 = self.feature_extraction(x2, self.backbone2)
        x3 = self.feature_extraction(x3, self.backbone3)
        x4 = self.feature_extraction(x4, self.backbone4)

        x1 = self.max_pool(x1)
        x2 = self.max_pool(x2)
        x3 = self.max_pool(x3)
        x4 = self.max_pool(x4)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)

        ga = self.gender_age(ga)
        x1 = self.FDR(x1) + ga
        x2 = self.FDR(x2) + ga
        x3 = self.FDR(x3) + ga
        x4 = self.FDR(x4) + ga

        x = torch.cat((x1.view(x1.size(0), 1, -1), x2.view(x1.size(0), 1, -1), x3.view(x1.size(0), 1, -1), x4.view(x1.size(0), 1, -1)), dim=1)
        x, _ = self.LSTM(x)

        x = self.mp(x.permute(0, 2, 1)).view(x.size(0), -1)
        x = self.last_linear(x)
        return x


class ResNet18_3D_4stream_clinical_LSTM_latefusion_fcn(nn.Module):
    def __init__(self, in_channels, n_classes, clinical_inchannels, pretrained=True, no_cuda=False, skip=False, pred_class=None):
        super(ResNet18_3D_4stream_clinical_LSTM_latefusion_fcn, self).__init__()

        self.classifer_p = ResNet18_3D_4stream_clinical_LSTM(in_channels, n_classes[0], clinical_inchannels, pretrained, no_cuda)
        self.classifer_m = ResNet18_3D_4stream_clinical_LSTM(in_channels, n_classes[1], clinical_inchannels, pretrained, no_cuda)

        self.classifer_p.load_state_dict(torch.load("/home/zky/ALMSS-main/pretrain_weight/primary_tumor.pth"))
        self.classifer_m.load_state_dict(torch.load("/home/zky/ALMSS-main/pretrain_weight/metastatic_tumor.pth"))

        self.skip = skip
        # self.attention = Transformer(dim=512, depth=3, heads=[2, 4, 8], mlp_dim=512)
        # self.mp = nn.MaxPool1d(8, stride=1)
        self.mp = nn.MaxPool1d(4, stride=1)
        if not pred_class:
            pred_class = n_classes[0]+n_classes[1]-1
        # self.last_linear = nn.Linear(512, n_classes[0]+n_classes[1]-1)
        self.last_linear = nn.Linear(512*2, pred_class)

    def feature_extraction(self, x, backbone):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        return x

    def forward(self, x1, x2, x3, x4, ga):
        x1_p = self.classifer_p.feature_extraction(x1, self.classifer_p.backbone1)
        x2_p = self.classifer_p.feature_extraction(x2, self.classifer_p.backbone2)
        x3_p = self.classifer_p.feature_extraction(x3, self.classifer_p.backbone3)
        x4_p = self.classifer_p.feature_extraction(x4, self.classifer_p.backbone4)
        x1_p = self.classifer_p.max_pool(x1_p)
        x2_p = self.classifer_p.max_pool(x2_p)
        x3_p = self.classifer_p.max_pool(x3_p)
        x4_p = self.classifer_p.max_pool(x4_p)
        x1_p = x1_p.view(x1_p.size(0), -1)
        x2_p = x2_p.view(x2_p.size(0), -1)
        x3_p = x3_p.view(x3_p.size(0), -1)
        x4_p = x4_p.view(x4_p.size(0), -1)
        ga_p = self.classifer_p.gender_age(ga)
        x1_p = self.classifer_p.FDR(x1_p) + ga_p
        x2_p = self.classifer_p.FDR(x2_p) + ga_p
        x3_p = self.classifer_p.FDR(x3_p) + ga_p
        x4_p = self.classifer_p.FDR(x4_p) + ga_p
        x_p = torch.cat((x1_p.view(x1_p.size(0), 1, -1), x2_p.view(x1_p.size(0), 1, -1), x3_p.view(x1_p.size(0), 1, -1), x4_p.view(x1_p.size(0), 1, -1)), dim=1)
        x_p, _ = self.classifer_p.LSTM(x_p)


        x1_m = self.classifer_m.feature_extraction(x1, self.classifer_m.backbone1)
        x2_m = self.classifer_m.feature_extraction(x2, self.classifer_m.backbone2)
        x3_m = self.classifer_m.feature_extraction(x3, self.classifer_m.backbone3)
        x4_m = self.classifer_m.feature_extraction(x4, self.classifer_m.backbone4)
        x1_m = self.classifer_m.max_pool(x1_m)
        x2_m = self.classifer_m.max_pool(x2_m)
        x3_m = self.classifer_m.max_pool(x3_m)
        x4_m = self.classifer_m.max_pool(x4_m)
        x1_m = x1_m.view(x1_m.size(0), -1)
        x2_m = x2_m.view(x2_m.size(0), -1)
        x3_m = x3_m.view(x3_m.size(0), -1)
        x4_m = x4_m.view(x4_m.size(0), -1)
        ga_m = self.classifer_m.gender_age(ga)
        x1_m = self.classifer_m.FDR(x1_m) + ga_m
        x2_m = self.classifer_m.FDR(x2_m) + ga_m
        x3_m = self.classifer_m.FDR(x3_m) + ga_m
        x4_m = self.classifer_m.FDR(x4_m) + ga_m
        x_m = torch.cat((x1_m.view(x1_m.size(0), 1, -1), x2_m.view(x1_m.size(0), 1, -1), x3_m.view(x1_m.size(0), 1, -1), x4_m.view(x1_m.size(0), 1, -1)), dim=1)
        x_m, _ = self.classifer_m.LSTM(x_m)

        x = torch.cat((x_p, x_m), dim=2)
        # if self.skip:
        #     x = x + self.attention(x)
        # else:
        #     x = self.attention(x)

        x = self.mp(x.permute(0, 2, 1)).view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def extract_feature(self, x1, x2, x3, x4, ga):
        x1_p = self.classifer_p.feature_extraction(x1, self.classifer_p.backbone1)
        x2_p = self.classifer_p.feature_extraction(x2, self.classifer_p.backbone2)
        x3_p = self.classifer_p.feature_extraction(x3, self.classifer_p.backbone3)
        x4_p = self.classifer_p.feature_extraction(x4, self.classifer_p.backbone4)
        x1_p = self.classifer_p.max_pool(x1_p)
        x2_p = self.classifer_p.max_pool(x2_p)
        x3_p = self.classifer_p.max_pool(x3_p)
        x4_p = self.classifer_p.max_pool(x4_p)
        x1_p = x1_p.view(x1_p.size(0), -1)
        x2_p = x2_p.view(x2_p.size(0), -1)
        x3_p = x3_p.view(x3_p.size(0), -1)
        x4_p = x4_p.view(x4_p.size(0), -1)
        ga_p = self.classifer_p.gender_age(ga)
        x1_p = self.classifer_p.FDR(x1_p) + ga_p
        x2_p = self.classifer_p.FDR(x2_p) + ga_p
        x3_p = self.classifer_p.FDR(x3_p) + ga_p
        x4_p = self.classifer_p.FDR(x4_p) + ga_p
        x_p = torch.cat((x1_p.view(x1_p.size(0), 1, -1), x2_p.view(x1_p.size(0), 1, -1), x3_p.view(x1_p.size(0), 1, -1), x4_p.view(x1_p.size(0), 1, -1)), dim=1)
        x_p, _ = self.classifer_p.LSTM(x_p)


        x1_m = self.classifer_m.feature_extraction(x1, self.classifer_m.backbone1)
        x2_m = self.classifer_m.feature_extraction(x2, self.classifer_m.backbone2)
        x3_m = self.classifer_m.feature_extraction(x3, self.classifer_m.backbone3)
        x4_m = self.classifer_m.feature_extraction(x4, self.classifer_m.backbone4)
        x1_m = self.classifer_m.max_pool(x1_m)
        x2_m = self.classifer_m.max_pool(x2_m)
        x3_m = self.classifer_m.max_pool(x3_m)
        x4_m = self.classifer_m.max_pool(x4_m)
        x1_m = x1_m.view(x1_m.size(0), -1)
        x2_m = x2_m.view(x2_m.size(0), -1)
        x3_m = x3_m.view(x3_m.size(0), -1)
        x4_m = x4_m.view(x4_m.size(0), -1)
        ga_m = self.classifer_m.gender_age(ga)
        x1_m = self.classifer_m.FDR(x1_m) + ga_m
        x2_m = self.classifer_m.FDR(x2_m) + ga_m
        x3_m = self.classifer_m.FDR(x3_m) + ga_m
        x4_m = self.classifer_m.FDR(x4_m) + ga_m
        x_m = torch.cat((x1_m.view(x1_m.size(0), 1, -1), x2_m.view(x1_m.size(0), 1, -1), x3_m.view(x1_m.size(0), 1, -1), x4_m.view(x1_m.size(0), 1, -1)), dim=1)
        x_m, _ = self.classifer_m.LSTM(x_m)

        x = torch.cat((x_p, x_m), dim=2)
        # if self.skip:
        #     x = x + self.attention(x)
        # else:
        #     x = self.attention(x)

        x = self.mp(x.permute(0, 2, 1)).view(x.size(0), -1)
        x = self.last_linear(x)
        return x





