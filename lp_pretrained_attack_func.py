import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import random

from generator_mask import Generator
from discriminator import Discriminator
from pixel_valuation import PVRL

models_path = "./models/lp_sep/"

class PVRL_Attack:
    def __init__(self,
                 device,
                 model,
                 generator,
                 discriminator,
                 model_num_labels,
                 box_min,
                 box_max):
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.box_min = box_min
        self.box_max = box_max

        self.netG = generator
        self.netDisc = discriminator
        self.netPv = PVRL().to(device)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=1e-3)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=1e-3)
        self.optimizer_PV = torch.optim.Adam(self.netPv.parameters(),
                                            lr=1e-2)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        
        self.optimizer_PV.zero_grad()
        
        # optimize D
        for i in range(1):
            # Mask
            probs = self.netPv(x)
            probs_cpu = probs.cpu().detach()
            probs_np = probs_cpu.numpy()
            mask = np.random.binomial(1, probs_np, probs_np.shape)
            mask = torch.from_numpy(mask)
            mask = mask.view(mask.size(0), 1, x.size(2), x.size(3))
            mask = mask.type(torch.FloatTensor).to(self.device)

            # Probs loss + Mask num pixels
            epsilon = 1e-8
            probs_arr = probs.view(probs.size(0), -1)
            mask_arr = mask.view(mask.size(0), -1)
            probs_loss = torch.sum(torch.log(probs_arr + epsilon) * mask_arr, dim=1) + torch.sum(torch.log(1 -probs_arr + epsilon) * (1 - mask_arr), dim=1)
            mask_num_pixels = torch.sum(mask_arr, dim=1)
            
            # Add mask as fourth channel + Generate perturbation
            x_with_mask = torch.cat((x, mask), 1).to(self.device)
            perturbation = self.netG(x_with_mask)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) * mask + x
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
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1) * mask.view(mask.shape[0], -1), 2, dim=1))
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
            loss_adv = torch.sum(loss_adv_arr)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward(retain_graph=True)
            self.optimizer_G.step()
        
        average_pixels = torch.mean(mask_num_pixels)
        rl_loss = torch.mean(-probs_loss * loss_adv_arr + torch.log(torch.sum(mask_num_pixels)))
        rl_loss.backward()
        self.optimizer_PV.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), rl_loss.item(), average_pixels.item()

    def train(self, train_dataloader, epochs):
        writer = SummaryWriter(log_dir='./visualization/lp_rl/sep/', comment='Limited Pixel Attack using Reinforcement Learning')
        for epoch in range(1, epochs+1):
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_rl_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_rl_batch, average_pixels = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_rl_sum += loss_rl_batch

            num_batch = len(train_dataloader)
            
            # print statistics
            print("epoch %d:\nloss_D: %.5f, loss_G_fake: %.5f,\
             \nloss_perturb: %.5f, loss_adv: %.5f, \nloss_rl: %.5f, \nAverage pixels: %.5f\n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch,
                    loss_rl_sum/num_batch, average_pixels))

            writer.add_scalar('advgan_discriminator_loss', loss_D_sum/num_batch, epoch)
            writer.add_scalar('advgan_generator_loss', loss_G_fake_sum/num_batch, epoch)
            writer.add_scalar('advgan_perturbation_loss', loss_perturb_sum/num_batch, epoch)
            writer.add_scalar('advgan_adversarial_loss', loss_adv_sum/num_batch, epoch)
            writer.add_scalar('pixel_valuation_loss', loss_rl_sum/num_batch, epoch)
            writer.add_scalar('average_num_pixels', average_pixels, epoch)

            netG_file_name = models_path + 'netG_rl_epoch_' + str(epoch) + '.pth'
            torch.save(self.netG.state_dict(), netG_file_name)
            netDisc_file_name = models_path + 'netDisc_rl_epoch_' + str(epoch) + '.pth'
            torch.save(self.netDisc.state_dict(), netDisc_file_name)
            netPv_file_name = models_path + 'netPv_rl_epoch_' + str(epoch) + '.pth'
            torch.save(self.netPv.state_dict(), netPv_file_name)
        writer.close()