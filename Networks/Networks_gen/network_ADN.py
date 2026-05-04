import torch
# import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import init
import functools
# from torch.optim import lr_scheduler
from Networks.Networks_gen.Swin_Unet_s_ACDC_2laterdown import BasicLayer1
# # from models.vit import Transformer
# from einops.layers.torch import Rearrange
# from einops import rearrange, repeat
# # import ml_collections
# import pywt
# import ml_collections
# from Contrast_Methods.models.residual_transformers import ResViT_3D_for_merge

from copy import deepcopy, copy
bias_setting = False


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss



def define_G(input_ch, base_ch, output_ch,netG,num_down, num_up,n_downsampling=3, resolution=[], load_path_pretrain=None,init_type='normal', init_gain=0.02, gpu=[], istrain=True):
    if netG == 'Swin_ADN':  #stage1
        net = CECT_ADN(input_ch=input_ch, base_ch=base_ch, output_ch=output_ch, num_down=num_down, num_up=num_up,n_downsampling=n_downsampling, resolution=resolution)
        if load_path_pretrain:
            pretrain_state = torch.load(load_path_pretrain, map_location='cpu')
            pretrained_netG_dict = pretrain_state['netG_state_dict']
            net.load_state_dict(pretrained_netG_dict, strict=True)

    elif netG == "Pre_Swin_ADN_10":  # stage2
        net = Pre_ADN_10(input_ch=input_ch, base_ch=base_ch, output_ch=output_ch, num_down=num_down, num_up=num_up,
                         n_downsampling=n_downsampling, resolution=resolution,
                         load_path_pretrain=load_path_pretrain, istrain=istrain)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu)


def define_D(input_ch, base_ch, output_ch=0, netD=None,num_down=3, num_up=3,n_downsampling=3, resolution=[], load_path_pretrain=None,init_type='normal', init_gain=0.02, gpu=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the networks.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu (int list) -- which GPUs the networks runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70脳70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    if netD == 'basic':  # default PatchGAN classifier
        net = NLayer3DDiscriminator(input_ch, ndf=base_ch, n_layers=3, norm_layer=nn.BatchNorm3d)
    elif netD == 'new_train':
        net = NLayer3DDiscriminator_art1(input_ch=input_ch, base_ch=base_ch, output_ch=output_ch, num_down=num_down, num_up=num_up,
                        n_downsampling=n_downsampling, resolution=resolution, load_path_pretrain=load_path_pretrain)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu)

class CECT_ADN(nn.Module):#预训练模型
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self, input_ch=1, base_ch=16, output_ch=1,num_down=3, num_up=3,n_downsampling=3, resolution=[],num_residual=4, num_sides="all",shared_decoder=False):
        super(CECT_ADN, self).__init__()
        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoder_art = Encoder(input_ch, base_ch, num_down)
        self.encoder_high = Encoder(input_ch, base_ch, num_down)
        self.decoder = Decoder(base_ch, output_ch,  num_up, num_residual, self.n, n_downsampling, resolution)
        self.decoder_art = self.decoder if shared_decoder else deepcopy(self.decoder)

    def forward1(self, x_low):
        code, sides = self.encoder_art(x_low)  # 增强CT->内容，增强成分
        y1 = self.decoder_art(code, sides[-self.n:]) # 重建增强CT
        y2 = self.decoder(code) #内容->平扫
        return y1, y2 # 增强CT，平扫CT

    def forward2(self, x_low, x_high):
        _, sides = self.encoder_art(x_low)  # encode artifact

        code, _ = self.encoder_high(x_high)  # encode high quality image
        y1 = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
        y2 = self.decoder(code)  # decode without artifact (high quality)
        return y1, y2  # 增强CT，平扫CT

    # def forward_lh(self, x_low):
    #     code, _ = self.encoder_low(x_low)  # encode low quality image
    #     y = self.decoder(code)
    #     return y

    def forward_hl(self, x_low, x_high):
        _, sides = self.encoder_art(x_low)  # encode artifact
        code, _ = self.encoder_high(x_high)  # encode high quality image
        y = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
        return y

class Pre_ADN_10(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self, input_ch=1, base_ch=16, output_ch=1,num_down=3, num_up=3,n_downsampling=3, resolution=[],load_path_pretrain=None,num_residual=4, num_sides="all",shared_decoder=False,istrain=True):
        super(Pre_ADN_10, self).__init__()

        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.pre_ADN = CECT_ADN(input_ch=input_ch, base_ch=base_ch, output_ch=output_ch,num_down=num_down, num_up=num_up,n_downsampling=n_downsampling, resolution=resolution)

        load_path = load_path_pretrain
        activation = nn.ReLU(True)
        norm_layer = nn.InstanceNorm3d

        self.patch_embedding0 = nn.Conv3d(in_channels=1, out_channels=base_ch*(2**(num_down)),
                                          kernel_size=(3, 3, 3),
                                          stride=(2**(num_down-1),2**(num_down),2**(num_down)), padding=(1, 1, 1))
        self.patch_embedding2 = nn.Conv3d(in_channels=1, out_channels=base_ch * 4,
                                          kernel_size=(3, 3, 3),
                                          stride=(2, 4, 4), padding=(1, 1, 1))
        self.patch_embedding1 = nn.Conv3d(in_channels=1, out_channels=base_ch*2,
                                          kernel_size=(3, 3, 3),
                                          stride=(1, 2, 2), padding=(1, 1, 1))
        self.patch_embedding3 = nn.Conv3d(in_channels=1, out_channels=base_ch * 8,
                                          kernel_size=(3, 3, 3),
                                          stride=(4, 8, 8), padding=(1, 1, 1))
        self.swinblock3=SwinResidualBlock_b2(base_ch * 8, dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.swinblock2=SwinResidualBlock_b1(base_ch * 4, dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.swinblock1=SwinResidualBlock_b0(base_ch*2, dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.res=SwinResidualBlock3d3(base_ch*(2**num_down), dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.res0=SwinResidualBlock3d3(base_ch*(2**num_down), dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.res1=SwinResidualBlock3d3(base_ch*(2**num_down), dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.res2=SwinResidualBlock3d3(base_ch*(2**num_down), dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        self.res3=SwinResidualBlock3d3(base_ch*(2**num_down), dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                     n_downsampling=n_downsampling,
                                     last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution)
        # if isinstance(net, torch.nn.DataParallel):
        #     net = net.module
        if istrain and load_path:
            print('编码器和解码器loading the model from %s' % load_path)
            state = torch.load(load_path, map_location='cpu')
            pretrained_netG_dict = state['netG_state_dict']
            model_netG_dict = self.pre_ADN.state_dict()
            pretrained_netG_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_netG_dict}
            model_netG_dict.update(pretrained_netG_dict)
            self.pre_ADN.load_state_dict(model_netG_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

        self.decoder_art = self.pre_ADN.decoder_art
        self.encoder_art = self.pre_ADN.encoder_art

    def forward(self, x_low):
        code, sides = self.encoder_art(x_low)

        x3=self.patch_embedding3(x_low)
        x3=self.swinblock3(x3)
        sides[0]=x3+sides[0]

        x2=self.patch_embedding2(x_low)
        x2= self.swinblock2(x2)
        sides[1]=x2+sides[1]

        x1=self.patch_embedding1(x_low)
        x1 = self.swinblock1(x1)
        sides[2]=x1+sides[2]

        x0 = self.patch_embedding0(x_low)
        x=self.res(x0)

        code=code+x

        y = self.decoder_art(code, sides)
        return y


class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down):
        super(Encoder, self).__init__()

        activation = nn.ReLU(True)
        conv0 = [nn.Conv3d(input_ch, base_ch, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 1, 1)),
                 activation]
        self.conv0 = nn.Sequential(*conv0)
        output_ch = base_ch * 2
        self.conv1 = De_Block(base_ch, output_ch, (3, 3, 3), (1, 1, 1), (1, 2, 2))
        output_ch *= 2
        base_ch *= 2
        self.conv2=De_Block(base_ch, output_ch, (3, 3, 3), (1, 1, 1), (2, 2, 2))
        output_ch *= 2
        base_ch *= 2
        self.conv3 = De_Block(base_ch, output_ch, (3, 3, 3), (1, 1, 1), (2, 2, 2))
        layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)]
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        sides = []
        x1=self.conv0(x)
        sides.append(x1)
        x1 = self.conv1(x1)
        sides.append(x1)
        x1=self.conv2(x1)
        sides.append(x1)
        x=self.conv3(x1)
        sides.append(x)
        return x, sides[::-1]#[::-1]元素按相反的顺序排列


class Decoder(nn.Module):
    def __init__(self,base_ch,  output_ch, num_up, num_residual, num_sides, n_downsampling, resolution,
                 norm_layer=nn.InstanceNorm3d, fuse=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** (num_up)
        input_chs = []
        activation = nn.ReLU(True)

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    SwinResidualBlock3d3(input_ch, dilation=[1, 2, 2], activation=activation, norm_layer=norm_layer,
                                         n_downsampling=n_downsampling,
                                         last_window_size=[[2, 4, 4]], last_num_heads=[8], resolution=resolution))
            input_chs.append(input_ch)
        for i in range(num_up-1):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                De_Block(input_ch, input_ch // 2, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2
        m = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode="nearest"),
            De_Block(input_ch, input_ch // 2, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
        setattr(self, "conv{}".format(num_up-1), m)
        input_chs.append(input_ch)
        input_ch //= 2
        m = Lastconvolution(base_ch, output_ch, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
                      [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # If true, fuse (concat and conv) the side features with decoder features
        # Otherwise, directly add artifact feature with decoder features
        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                        nn.Conv3d(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            self.fuse = lambda x, y, i: x + y

    def forward(self, x, sides=[]):
        m, n = len(self.layers), len(sides)#m=8,n=4

        assert m >= n, "Invalid side inputs"

        for i in range(m-n):
            x = self.layers[i](x)#(1,128,2,32,32)

        for i, j in enumerate(range(m-n, m)):#4,5,6,7
            x = self.fuse(x, sides[i], i)
            x = self.layers[j](x)#(1,128,2,32,32)-4>(1,64,4,64,64)-5>(1,32,8,128,128)-6>(1,16,8,256,256)-7>(1,1,8,256,256)

        return x

class SwinResidualBlock3d3(nn.Module):
    def __init__(self, dim, dilation, norm_layer, n_downsampling, last_window_size, last_num_heads, resolution,
                 activation=nn.ReLU(True)):
        super(SwinResidualBlock3d3, self).__init__()

        depths = [2]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = len(last_num_heads)

        self.last_patch_embeddings = nn.Conv3d(in_channels=self.last_hidden_size, out_channels=self.last_hidden_size,
                                               kernel_size=1,
                                               stride=1)

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer1(
                dim=self.last_hidden_size,
                input_resolution=(
                    int(resolution[0] / (2 ** (n_downsampling-1))), int(resolution[1] / (2 ** (n_downsampling))),
                    int(resolution[2] / (2 ** (n_downsampling)))),
                depth=depths[0],
                num_heads=last_num_heads[i_layer],
                window_size=last_window_size[i_layer],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)

    def forward(self, x):
        x_orgin = x

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        x_shout = x
        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)
            x = x + x_shout
            x_shout = x

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()

        out = x + x_orgin

        return out


class SwinResidualBlock_b0(nn.Module):
    def __init__(self, dim, dilation, norm_layer, n_downsampling, last_window_size, last_num_heads, resolution,
                 activation=nn.ReLU(True)):
        super(SwinResidualBlock_b0, self).__init__()

        depths = [1]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = len(last_num_heads)

        self.last_patch_embeddings = nn.Conv3d(in_channels=self.last_hidden_size, out_channels=self.last_hidden_size,
                                               kernel_size=1,
                                               stride=1)

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer1(
                dim=self.last_hidden_size,
                input_resolution=(
                    int(resolution[0] ), int(resolution[1] / 2),
                    int(resolution[2] / 2)),
                depth=depths[0],
                num_heads=last_num_heads[i_layer],
                window_size=last_window_size[i_layer],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)

    def forward(self, x):
        x_orgin = x

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        x_shout = x
        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)
            x = x + x_shout
            x_shout = x

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()

        out = x + x_orgin

        return out


class SwinResidualBlock_b1(nn.Module):
    def __init__(self, dim, dilation, norm_layer, n_downsampling, last_window_size, last_num_heads, resolution,
                 activation=nn.ReLU(True)):
        super(SwinResidualBlock_b1, self).__init__()

        depths = [2]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = len(last_num_heads)

        self.last_patch_embeddings = nn.Conv3d(in_channels=self.last_hidden_size, out_channels=self.last_hidden_size,
                                               kernel_size=1,
                                               stride=1)

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer1(
                dim=self.last_hidden_size,
                input_resolution=(
                    int(resolution[0] / 2) , int(resolution[1] / (2 ** (n_downsampling-1))),
                    int(resolution[2] / (2 ** (n_downsampling-1)))),
                depth=depths[0],
                num_heads=last_num_heads[i_layer],
                window_size=last_window_size[i_layer],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)

    def forward(self, x):
        x_orgin = x

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        x_shout = x
        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)
            x = x + x_shout
            x_shout = x

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()

        out = x + x_orgin

        return out

class SwinResidualBlock_b2(nn.Module):
    def __init__(self, dim, dilation, norm_layer, n_downsampling, last_window_size, last_num_heads, resolution,
                 activation=nn.ReLU(True)):
        super(SwinResidualBlock_b2, self).__init__()

        depths = [2]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = len(last_num_heads)

        self.last_patch_embeddings = nn.Conv3d(in_channels=self.last_hidden_size, out_channels=self.last_hidden_size,
                                               kernel_size=1,
                                               stride=1)

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer1(
                dim=self.last_hidden_size,
                input_resolution=(
                    int(resolution[0] / (2 ** (n_downsampling-1))), int(resolution[1] / (2 ** (n_downsampling))),
                    int(resolution[2] / (2 ** (n_downsampling)))),
                depth=depths[0],
                num_heads=last_num_heads[i_layer],
                window_size=last_window_size[i_layer],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)

    def forward(self, x):
        x_orgin = x

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        x_shout = x
        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)
            x = x + x_shout
            x_shout = x

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()

        out = x + x_orgin

        return out



class NLayer3DDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayer3DDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1,2,2), padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1,2,2), padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, isDetach):
        """Standard forward."""
        return self.model(input)


class NLayer3DDiscriminator_art1(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_ch=1, base_ch=16, output_ch=1,num_down=3, num_up=3,n_downsampling=3, resolution=[],load_path_pretrain = None,num_residual=4, num_sides="all",shared_decoder=False):
        super(NLayer3DDiscriminator_art1, self).__init__()

        self.pre_ADN = CECT_ADN(input_ch=input_ch, base_ch=base_ch, output_ch=output_ch,num_down=num_down, num_up=num_up,n_downsampling=n_downsampling, resolution=resolution)

        load_path = load_path_pretrain
        if load_path != '':
            print('using pretrain_D,loading the model from %s' % load_path)
            # state = torch.load(load_path)
            state = torch.load(load_path, map_location='cpu')
            pretrained_netG_dict = state['netG_state_dict']
            model_netG_dict = self.pre_ADN.state_dict()
            pretrained_netG_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_netG_dict}
            model_netG_dict.update(pretrained_netG_dict)
            self.pre_ADN.load_state_dict(model_netG_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中
        self.encoder=self.pre_ADN.encoder_art
        new_conv=nn.Conv3d(in_channels=base_ch*8, out_channels=1, kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        layers = list(self.encoder.layers)
        layers.append(new_conv)
        new_model = nn.Sequential(*layers)
        self.encoder = new_model
        # self.encoder = Encoder(input_ch, base_ch, num_down)
        # self.decoder = Decoder(base_ch, output_ch, num_up, num_residual, self.n, n_downsampling, resolution)
        # self.last_layer0 = Lastconvolution(output_ch*2, output_ch, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def forward(self, input,isDetach):
        """Standard forward."""
        return self.encoder(input)

def init_net(net, init_type='normal', init_gain=0.02, gpu=[]):
    """Initialize a networks: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the networks weights
    Parameters:
        net (networks)      -- the networks to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu (int list) -- which GPUs the networks runs on: e.g., 0,1,2

    Return an initialized networks.
    """

    assert (torch.cuda.is_available())
    net.to(gpu)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1) and (
                classname != 'ShareSepConv3d' or classname != 'ShareSepConv2d'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            print('Norm initialized')
            init.normal(m.weight.data, 1.0, init_gain)
            init.constant(m.bias.data, 0.0)

    print('initialize networks with %s' % init_type)
    net.apply(init_func)

class De_Block(nn.Module):
    def __init__(self, input_nc, ngf, num_kernel, padw, strides, norm_layer=nn.InstanceNorm3d,
                 activation=nn.ReLU(True)):
        super(De_Block, self).__init__()
        conv0 = [nn.Conv3d(input_nc, ngf, kernel_size=num_kernel, padding=padw, stride=strides, bias=bias_setting),
                 norm_layer(ngf),
                 activation]
        self.conv = nn.Sequential(*conv0)

    def forward(self, x):
        out = self.conv(x)
        return out


class Lastconvolution(nn.Module):
    def __init__(self, input_nc, ngf, num_kernel, padw, strides, activation=nn.Tanh()):
        super(Lastconvolution, self).__init__()
        conv = [nn.Conv3d(input_nc, ngf, kernel_size=num_kernel, padding=padw, stride=strides, bias=bias_setting),
                activation]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

