import os
import torch
from . import network_ADN
import torch.nn as nn

def update_learning_rate(model, max_epochs, epoch, lr_max ):
    """Update learning rates for all the networks; called at the end of every epoch"""
    # for scheduler in self.schedulers:
    #     scheduler.step()
    model.optimizer_G.param_groups[0]['lr'] = lr_max * (1 - epoch / max_epochs) ** 0.998
    model.optimizer_D.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']

def save_networks(opt, save_name, model, epoch):

    save_filename = '%s.pth' % (save_name)
    save_path = os.path.join(opt.model_results, save_filename)

    state = {
        'epoch': epoch + 1,
        'netG_state_dict': model.netG.state_dict(),
        'netD_state_dict': model.netD.state_dict(),
        'optimizer_G': model.optimizer_G.state_dict(),
        'optimizer_D': model.optimizer_D.state_dict(),
    }

    torch.save(state, save_path)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (networks list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_networks(opt, model):
    load_filename = '%s.pth' % (opt.load_name)
    load_path = os.path.join(opt.model_results, load_filename)

    print('loading the model from %s' % load_path)

    try:
        state = torch.load(load_path, map_location='cuda:0')
    except:
        state = torch.load(opt.load_name, map_location='cuda:0')

    pretrained_netG_dict = state['netG_state_dict']

    model_netG_dict = model.netG.state_dict()
    # pretrained_netG_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_netG_dict}

    model_netG_dict.update(pretrained_netG_dict)
    model.netG.load_state_dict(model_netG_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中
    if opt.isTrain:
        pretrained_netD_dict = state['netD_state_dict']
        model_netD_dict = model.netD.state_dict()
        pretrained_netD_dict = {k: v for k, v in pretrained_netD_dict.items() if k in model_netD_dict}
        model_netD_dict.update(pretrained_netD_dict)
        model.netD.load_state_dict(model_netD_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

        model.optimizer_G.load_state_dict(state['optimizer_G'])
        model.optimizer_D.load_state_dict(state['optimizer_D'])

    opt.epoch_count = state['epoch']
    print('Successfully loading the model from %s' % load_path)

class Pix2Pix3DModel_pretrain(nn.Module):
    def __init__(self, opt):
        super(Pix2Pix3DModel_pretrain, self).__init__()
        self.isTrain = opt.isTrain
        self.resolution = [opt.depthSize, opt.ImageSize_x, opt.ImageSize_y]
        self.gpu = opt.gpu
        self.device = opt.device
        self.lambda_L1 = opt.lambda_L1
        self.l1_weight_inmask = opt.l1_weight_inmask
        # self.l1_weight_wholeimage = 1-opt.l1_weight_inmask
        self.l1_weight_wholeimage = 1
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_loss']
        self.netG = network_ADN.define_G(input_ch=opt.input_nc, base_ch=opt.ngf, output_ch=opt.output_nc,
                                         netG=opt.G_model, num_down=opt.num_down, num_up=opt.num_up,
                                         n_downsampling=opt.n_downsampling, resolution=self.resolution,
                                         load_path_pretrain=opt.load_path_pretrain, init_type=opt.init_type,
                                         init_gain=opt.init_gain, gpu=opt.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = network_ADN.define_D(input_ch=opt.input_nc + opt.output_nc, base_ch=opt.ndf, netD=opt.D_model,
                                             init_type=opt.init_type, init_gain=opt.init_gain, gpu=self.device)
            self.netD1 = network_ADN.define_D(input_ch=opt.input_nc + opt.output_nc, base_ch=opt.ndf, netD=opt.D_model,
                                             init_type=opt.init_type, init_gain=opt.init_gain, gpu=self.device)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = network_ADN.GANLoss(opt.gan_mode).to(self.device)

            self.criterion_L1 = torch.nn.L1Loss().to(self.device)

            #self.criterionPreLoss = VGGLoss_3D(opt.pretrain_model_path).to(self.device)
            # self.Charbonnier_loss = L1_Charbonnier_loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))


    def set_input(self, input, sample_masktype='body_mask'):
        self.real_A = input['A'].to(self.device)  #增强CT
        self.real_B = input['B'].to(self.device)  #平扫CT
        self.mask = input[sample_masktype].to(self.device)


    def forward(self, epoch):
        # self.real_A = self.real_A.cuda()  #增强CT
        # self.real_B = self.real_B.cuda()  #平扫CT
        # self.mask =  self.mask.cuda()

        set_requires_grad(self.netD, True)
        set_requires_grad(self.netD1, True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.optimizer_D1.zero_grad()
        self.optimizer_G.zero_grad()
        # calculate gradients for D

        """增强图像成分拆分： 增强->增强,增强->平扫，判别合成平扫"""
        self.pred_lh, self.pred_ll=self.netG.forward1(self.real_A) #合成增强,合成平扫
        # Fake; stop backprop to the generator by detaching fake_B
        fake_BA = torch.cat((self.real_A, self.pred_ll), 1)
        pred_fake = self.netD(fake_BA.detach(), isDetach=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_BA=torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_BA, isDetach=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_loss = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D_loss.backward()
        self.optimizer_D.step()  # update D's weights

        """平扫->平扫,平扫->增强"""
        self.pred_hh, self.pred_hl = self.netG.forward2(self.real_A, self.real_B)
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_B, self.pred_hh), 1)
        pred_fake = self.netD1(fake_AB.detach(), isDetach=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_B, self.real_A), 1)
        pred_real = self.netD1(real_AB, isDetach=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_loss1 = (self.loss_D_fake + self.loss_D_real) * 0.5
        #
        # self.loss_G_hh.backward(retain_graph=True)
        self.loss_D_loss1.backward()
        self.optimizer_D1.step()  # update D's weights


        set_requires_grad(self.netD, False)
        set_requires_grad(self.netD1, False)
        # cycle loss
        """合成平扫->合成增强"""
        self.pred_lhh = self.netG.forward_hl(self.pred_hh, self.pred_ll)
        self.lossG_lhh = self.criterion_L1(self.pred_lhh, self.real_A) * self.l1_weight_wholeimage + self.criterion_L1(self.pred_lhh*self.mask, self.real_A*self.mask) * self.l1_weight_inmask
        """high_l->high_l_h"""
        self.lossG_art = self.criterion_L1(self.pred_hh-self.real_B, self.real_A-self.pred_ll) * self.l1_weight_wholeimage + self.criterion_L1((self.pred_hh-self.real_B)*self.mask, (self.real_A-self.pred_ll)*self.mask) * self.l1_weight_inmask
        self.lossG_art1 = self.criterion_L1(self.pred_lh, self.pred_hh) * self.l1_weight_wholeimage + self.criterion_L1(self.pred_lh*self.mask, self.pred_hh*self.mask) * self.l1_weight_inmask
        self.loss_G_ll = self.criterion_L1(self.pred_hl, self.real_B) * self.l1_weight_wholeimage + self.criterion_L1(self.pred_hl*self.mask, self.real_B*self.mask) * self.l1_weight_inmask  # 重建
        self.loss_G_hh = self.criterion_L1(self.pred_lh, self.real_A) * self.l1_weight_wholeimage + self.criterion_L1(self.pred_lh*self.mask, self.real_A*self.mask) * self.l1_weight_inmask  # 重建
        self.loss_G_lh = self.criterion_L1(self.pred_hh, self.real_A) * self.l1_weight_wholeimage + self.criterion_L1(self.pred_hh*self.mask, self.real_A*self.mask) * self.l1_weight_inmask  # regression L1

        fake_BA = torch.cat((self.real_A, self.pred_ll), 1)
        pred_fake = self.netD(fake_BA, isDetach=False)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        fake_AB = torch.cat((self.real_B, self.pred_hh), 1)
        pred_fake = self.netD1(fake_AB, isDetach=False)
        self.loss_G_GAN1 = self.criterionGAN(pred_fake, True)


        self.loss_g = ((0.4*self.loss_G_ll+self.loss_G_hh)+(self.loss_G_lh)+(self.lossG_lhh) + (self.lossG_art+self.lossG_art1))*self.lambda_L1 +self.loss_G_GAN+self.loss_G_GAN1  # zky
        self.loss_g.backward()
        self.optimizer_G.step()  # udpate G's weights

class Pix2Pix3DModel(nn.Module):
    def __init__(self, opt):
        super(Pix2Pix3DModel, self).__init__()
        self.l1_weight_inmask = opt.l1_weight_inmask
        self.l1_weight_wholeimage = 1
        # self.l1_weight_wholeimage = 1 - opt.l1_weight_inmask
        self.isTrain = opt.isTrain
        self.resolution = [opt.depthSize, opt.ImageSize_x, opt.ImageSize_y]
        self.gpu = opt.device
        self.device = opt.device
        # self.device_vgg = opt.device_vgg
        self.lambda_L1 = opt.lambda_L1
        self.lambda_L0 = opt.lambda_L0
        self.lambda_idt=opt.lambda_idt
        self.lambda_NCC=opt.lambda_NCC
        # self.type = opt.type
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_loss']
        self.netG = network_ADN.define_G(opt.input_nc, opt.ngf, opt.output_nc, opt.G_model,opt.num_down,opt.num_up,opt.n_downsampling,self.resolution,opt.load_path_pretrain,opt.init_type, opt.init_gain, self.gpu,istrain=self.isTrain)

        if self.isTrain:
            self.netD = network_ADN.define_D(opt.input_nc, opt.ngf, opt.output_nc, opt.D_model,opt.num_down,opt.num_up,opt.n_downsampling,self.resolution,opt.load_path_pretrain,opt.init_type, opt.init_gain, self.gpu)
            # self.netD = network_ADN.define_D1(opt.input_nc, opt.ngf, opt.output_nc, opt.D_model,opt.num_down,opt.num_up,opt.n_downsampling,self.resolution,opt.load_path_pretrain,opt.init_type, opt.init_gain, self.gpu)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network_ADN.GANLoss(opt.gan_mode).to(self.device)
            self.criterionidt = torch.nn.L1Loss().to(self.device)

            self.criterionNCC = network_ADN.Enhanced_loss_new(opt.input_nc, opt.ngf, opt.output_nc, opt.num_down,
                                                                  opt.num_up, opt.n_downsampling, self.resolution,opt.load_path_pretrain).to(self.device)

            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            # self.criterionVGGLoss = VGGLoss_3D(opt.pretrain_model_path).to(self.device_vgg)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))

    def set_input(self, input, sample_masktype='body_mask'):
        self.real_A = input['A'].to(self.device) #平扫
        self.real_B = input['B'].to(self.device) #增强
        self.mask = input[sample_masktype].to(self.device)

    def forward(self):
        # self.real_A = self.real_A.cuda() #平扫
        # self.real_B = self.real_B.cuda() #增强

        self.fake_B = self.netG(self.real_A)

        set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        # calculate gradients for D
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach(), isDetach=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_B, isDetach=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_loss = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D_loss.backward()
        self.optimizer_D.step()  # update D's weights
        # update G
        set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero

        # backward_G
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B, isDetach=False)
        self.fake_B0 = self.netG(self.real_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_idt = self.criterionidt(self.fake_B0, self.real_B) * self.lambda_idt
        self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B)*self.l1_weight_wholeimage +self.criterionL1(self.fake_B*self.mask, self.real_B*self.mask)*self.l1_weight_inmask)* self.lambda_L1
        self.loss_G_NCC = self.criterionNCC(self.fake_B, self.real_B) * self.lambda_NCC


        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_idt + self.loss_G_NCC

        self.loss_G.backward()
        self.optimizer_G.step()






