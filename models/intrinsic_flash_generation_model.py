import torch
from .base_model import BaseModel
from . import networksnetworks
import matplotlib.pyplot as plt
import numpy as np
import numpy
import itertools
from util.image_pool import ImagePool
from torchvision import transforms
import cv2
from models.altered_midas.flash_nets import DecomposeNet, GenerateNet
from PIL import Image, ImageDraw


class IntrinsicFlashGenerationModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_gradient', type=float, default=0)
        parser.add_argument('--no_vgg_loss', action='store_true')
        parser.add_argument('--no_cycle_loss', action='store_true')
        parser.add_argument('--no_gan_loss', action='store_true')
        parser.add_argument('--no_geometry', action='store_true')
        parser.add_argument('--no_gradient_loss', action='store_true')

        if is_train:
            # 100 for L1 and 25 for A and B
            parser.add_argument('--lambda_feat', type=float, default=40)
            parser.set_defaults(pool_size=0, gan_mode='wgangp')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_A', type=float, default=50.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=50.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--cycle_epoch', type=float, default=30, help='')
            parser.add_argument('--lambda_color', type=float, default=1)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_L1_gen', 'G_L1_dec', 'cycle_ambient', 'cycle_flashphoto', 'color']
        if not self.opt.no_vgg_loss:
            self.loss_names += ['VGG_dec', 'VGG_gen']
        if self.opt.lambda_gradient:
            self.loss_names += ['gradient_A', 'gradient_B']
        if not self.opt.no_gan_loss:
            self.loss_names += ['D_dec', 'D_gen', 'G_GAN_gen', 'G_GAN_dec']

        visual_names_B = ['fake_flashPhoto_gen', 'ambi_impl_shd']
        visual_names_A = ['fake_ambient_dec', "dec_impl_alb", "albedo_med"]
        visual_names_Sh = ['grnd_flsh_shd_gen', 'pred_ambi_shd_dec_colored', 'pred_flsh_shd_gen', 'albedo_amb']
        self.visual_names = visual_names_A + visual_names_B + visual_names_Sh
        self.colors = ['ambient_shading_color', "fake_ambient_color"]

        self.model_names = ['G_Decomposition', 'G_Generation']

        if self.isTrain and not self.opt.no_gan_loss:
            self.model_names += ['D_Decomposition', 'D_Generation']

        if self.opt.no_geometry:
            self.netG_Generation = GenerateNet(input_channels=3, activation='sigmoid')
            self.netG_Decomposition = DecomposeNet(input_channels=6, activation='sigmoid')
        else:
            self.netG_Generation = GenerateNet(input_channels=7, activation='sigmoid')
            self.netG_Decomposition = DecomposeNet(input_channels=10, activation='sigmoid')

        self.netG_Decomposition = networks.init_net(self.netG_Decomposition, gpu_ids=self.gpu_ids)
        self.netG_Generation = networks.init_net(self.netG_Generation, gpu_ids=self.gpu_ids)
        if self.isTrain and not self.opt.no_gan_loss:
            self.netD_Decomposition = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                        self.gpu_ids)
            self.netD_Generation = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                     self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt.netD).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_Decomposition.parameters(), self.netG_Generation.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            if not self.opt.no_gan_loss:
                self.optimizer_D = torch.optim.Adam(
                    itertools.chain(self.netD_Decomposition.parameters(), self.netD_Generation.parameters()),
                    lr=opt.lr2,
                    betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_flashPhoto = input['flashPhoto'].to(self.device)
        self.real_ambient = input['ambient'].to(self.device)
        self.image_paths = input['image_path']
        self.albedo_amb = input['albedo_amb'].to(self.device)
        self.albedo_flshpht = input['albedo_flshpht'].to(self.device)
        self.normals_amb = input['normals_amb'].to(self.device)
        self.normals_flshpht = input['normals_flshpht'].to(self.device)
        self.flsh_impl_shd_med = input['flsh_impl_shd_med'].to(self.device)
        self.flsh_impl_shd_amb = input['flsh_impl_shd_amb'].to(self.device)
        self.ambi_impl_shd = input['ambi_impl_shd'].to(self.device)
        self.albedo_med = input['albedo_med'].to(self.device)
        self.ambient_shading_temp = input['ambient_shading_temp'].to(self.device)
        self.depth_flashPhoto = input['depth_flashPhoto'].to(self.device)
        self.depth_ambient = input['depth_ambient'].to(self.device)
        self.real_flashPhoto_wb = input['flashPhoto_wb'].to(self.device)
        self.ambient_shading_color = input['ambient_shading_color']
        self.real_ambient_wb = input['ambient_wb'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.no_geometry:
            decomposition_input = torch.cat(
                (self.real_flashPhoto, self.albedo_flshpht), 1)
            generation_input = self.albedo_amb
        else:
            decomposition_input = torch.cat(
                (self.real_flashPhoto, self.depth_flashPhoto, self.albedo_flshpht, self.normals_flshpht), 1)
            generation_input = torch.cat((self.depth_ambient, self.albedo_amb, self.normals_amb), 1)

        # forward into networks
        self.fake_flash_shading_dec, self.fake_ambient_shading_dec, self.fake_ambient_temp_dec = self.netG_Decomposition(
            decomposition_input)
        self.fake_flash_shading_gen = self.netG_Generation(generation_input)

        # convert the shading predictions to the actual shading space
        dec_flsh_shd = (1.0 / self.fake_flash_shading_dec) - 1.0
        dec_ambi_shd = (1.0 / self.fake_ambient_shading_dec) - 1.0
        # color the ambient shading using the predicted ambient color temps
        self.fake_ambient_color, dec_ambi_shd_clr = self.color_ambient(dec_ambi_shd, self.fake_ambient_temp_dec)
        # compute the new implied albedo from the input image and the predicted shading
        self.dec_impl_alb = self.real_flashPhoto / (dec_flsh_shd + dec_ambi_shd_clr)
        self.dec_ambi_shd_clr = 1.0 / (dec_ambi_shd_clr + 1.0)
        # compute the predicted ambient image from the implied alb and colored ambient shading
        self.fake_ambient_dec = dec_ambi_shd_clr * self.dec_impl_alb
        self.fake_ambient_dec_wb = dec_ambi_shd * self.dec_impl_alb
        self.fake_flashphoto_dec = self.fake_ambient_dec_wb + (dec_flsh_shd * self.dec_impl_alb)
        # generate the predicted flash photo from the input albedo and predicted flash shd
        gen_flsh_shd = (1.0 / self.fake_flash_shading_gen) - 1.0
        gen_flsh_shd = gen_flsh_shd * 1.5
        # self.gen_flsh_shd = self.color_flash(gen_flsh_shd)
        self.fake_flash_gen = self.albedo_amb * gen_flsh_shd
        self.fake_flash_dec = dec_flsh_shd * self.dec_impl_alb
        self.fake_flashPhoto_gen = (self.albedo_amb * gen_flsh_shd) + self.real_ambient_wb
        if not self.opt.no_cycle_loss:
            # ambient --> flash photo --> ambient
            # -------------------------------------------------
            if self.opt.no_geometry:
                decomposition_input_cycle = torch.cat(
                    (self.fake_flashPhoto_gen, self.albedo_flshpht), 1)
            else:
                decomposition_input_cycle = torch.cat(
                    (self.fake_flashPhoto_gen, self.depth_flashPhoto, self.albedo_flshpht, self.normals_flshpht), 1)
            self.rec_flash_shading_dec, self.rec_ambient_shading_dec, self.rec_ambient_temp_dec = self.netG_Decomposition(
                decomposition_input_cycle)

            # shadings computed from the generated flash photo
            rec_flsh_shd = (1.0 / self.rec_flash_shading_dec) - 1.0
            rec_ambi_shd = (1.0 / self.rec_ambient_shading_dec) - 1.0

            # color ambient shading from our generated flash photo
            _, rec_ambi_shd_clr = self.color_ambient(rec_ambi_shd, self.rec_ambient_temp_dec)

            # compute the implied albedo from our generated flash photo
            self.rec_impl_alb = self.fake_flashPhoto_gen / (rec_flsh_shd + rec_ambi_shd_clr)

            # compute the ambient image for our generated flash photo
            self.rec_ambient = rec_ambi_shd * self.rec_impl_alb
            # -------------------------------------------------

            # flash photo --> ambient --> flash photo
            # -------------------------------------------------
            if self.opt.no_geometry:
                generation_input_cycle = self.albedo_amb
            else:
                generation_input_cycle = torch.cat((self.depth_ambient, self.albedo_amb, self.normals_amb), 1)
            self.rec_flash_shading_gen = self.netG_Generation(generation_input_cycle)
            gen_flsh_shd = (1.0 / self.rec_flash_shading_gen) - 1.0
            self.rec_flashPhoto = (self.albedo_amb * gen_flsh_shd) + self.fake_ambient_dec
            # -------------------------------------------------
        self.pred_flsh_shd_gen = (1 - self.fake_flash_shading_gen) * 255
        self.grnd_flsh_shd_gen = (1 - self.flsh_impl_shd_med) * 255
        self.pred_flsh_shd_dec = (1 - self.fake_flash_shading_dec) * 255
        self.pred_ambi_shd_dec = (1 - self.dec_ambi_shd_clr)
        self.ambi_impl_shd = (1 - self.ambi_impl_shd) * 255

    def backward_D_basic(self, netD, real_input, real_output, fake_output):
        # Fake
        fake_input_output = torch.cat((real_input, fake_output), 1)
        pred_fake = netD(fake_input_output.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_input_output = torch.cat((real_input, real_output), 1)
        pred_real = netD(real_input_output)
        loss_D_real = self.criterionGAN(pred_real, True)

        if (self.opt.gan_mode == 'wgangp'):
            self.loss_gradient_penalty, gradients = networks.cal_gradient_penalty(
                netD, real_input_output, fake_input_output, self.device, lambda_gp=20.0
            )
            self.loss_gradient_penalty.backward(retain_graph=True)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_gen(self):
        self.loss_D_gen = self.backward_D_basic(self.netD_Generation, self.real_ambient, self.real_flashPhoto,
                                                self.fake_flashPhoto_gen)

    def backward_G(self, epoch):

        self.loss_cycle_flashphoto = 0
        self.loss_cycle_ambient = 0
        self.loss_G_GAN_gen = 0
        self.loss_G_GAN_dec = 0
        if not self.opt.no_gan_loss:
            fake_ambient = torch.cat((self.real_flashPhoto, self.fake_ambient_dec), 1)
            pred_fake = self.netD_Decomposition(fake_ambient)
            self.loss_G_GAN_gen = self.criterionGAN(pred_fake, True)

            fake_flashPhoto = torch.cat((self.real_ambient, self.fake_flashPhoto_gen), 1)
            pred_fake = self.netD_Generation(fake_flashPhoto)
            self.loss_G_GAN_dec = self.criterionGAN(pred_fake, True)

        self.loss_VGG_gen = 0
        self.loss_VGG_dec = 0

        # make sure the predictions and ground-truth are [-1, 1]
        if not self.opt.no_vgg_loss:
            self.loss_VGG_dec = self.criterionVGG(
                self.fake_ambient_dec,
                self.real_ambient
            ) * self.opt.lambda_feat

            self.loss_VGG_gen = self.criterionVGG(
                self.fake_flashPhoto_gen,
                self.real_flashPhoto
            ) * self.opt.lambda_feat
        if not self.opt.no_cycle_loss:
            self.loss_cycle_ambient = l1_grad_loss(self.real_ambient,
                                                   self.rec_ambient, self.opt.no_gradient_loss) * self.opt.lambda_A
            self.loss_cycle_flashphoto = l1_grad_loss(self.rec_flashPhoto, self.real_flashPhoto_wb,
                                                      self.opt.no_gradient_loss) * self.opt.lambda_B

        self.loss_G_L1_gen = l1_grad_loss(self.fake_flash_shading_gen, self.flsh_impl_shd_med,
                                          self.opt.no_gradient_loss) * self.opt.lambda_L1 + \
                             l1_grad_loss(self.fake_flashPhoto_gen, self.real_flashPhoto_wb,
                                          self.opt.no_gradient_loss) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1_gen + \
                      self.loss_VGG_dec + self.loss_VGG_gen + \
                      self.loss_G_GAN_gen + \
                      self.loss_cycle_flashphoto + self.loss_cycle_ambient

        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()  # compute fake images: G(A)
        if not self.opt.no_gan_loss:
            self.set_requires_grad([self.netD_Decomposition, self.netD_Generation], True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_gen()  # calculate gradients for D_A
            self.backward_D_dec()
            self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad([self.netD_Decomposition, self.netD_Generation],
                                   False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G(epoch)  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
