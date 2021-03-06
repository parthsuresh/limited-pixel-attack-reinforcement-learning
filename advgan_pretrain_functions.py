import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os

from orig_generator import Generator
from discriminator import Discriminator

models_path = "./models/adv_gan/"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Pretrain:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 box_min,
                 box_max):
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.box_min = box_min
        self.box_max = box_max

        self.netG = Generator().to(device)
        self.netDisc = Discriminator().to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=1e-3)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=1e-3)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv_arr = torch.max(real - other, zeros)
            print(loss_adv_arr)
            print(loss_adv_arr.shape)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        writer = SummaryWriter(log_dir="visualization/orig_advgan/", comment='Original AdvGAN stats')
        
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=1e-4)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=1e-4)
                
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=1e-5)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=1e-5)
                
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            writer.add_scalar('discriminator_loss', loss_D_sum/num_batch, epoch)
            writer.add_scalar('generator_loss', loss_G_fake_sum/num_batch, epoch)
            writer.add_scalar('perturbation_loss', loss_perturb_sum/num_batch, epoch)
            writer.add_scalar('adversarial_loss', loss_adv_sum/num_batch, epoch)
            print("epoch %d:\nloss_D: %.5f, loss_G_fake: %.5f,\
             \nloss_perturb: %.5f, loss_adv: %.5f\n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'netG_original_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
                netDisc_file_name = models_path + 'netDisc_original_epoch_' + str(epoch) + '.pth'
                torch.save(self.netDisc.state_dict(), netDisc_file_name)
        
        writer.close()