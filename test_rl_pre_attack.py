import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import os

from generator_mask import Generator
from discriminator import Discriminator
from pixel_valuation import PVRL
import resnet_model

use_cuda=True
batch_size = 128

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

images_path = "./images/"

if not os.path.exists(images_path):
    os.makedirs(images_path)

# load the pretrained model
checkpoint = torch.load("./save_temp/checkpoint.th")
targeted_model = torch.nn.DataParallel(resnet_model.__dict__['resnet32']())
targeted_model.cuda()
targeted_model.load_state_dict(checkpoint['state_dict'])
targeted_model.eval()

# load the generator of adversarial gan
pretrained_generator_path = './models/lp_pretrained/netG_rl_epoch_20.pth'
pretrained_G = Generator().to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# load the discriminator of adversarial gan
pretrained_disciminator_path = './models/lp_pretrained/netDisc_rl_epoch_20.pth'
pretrained_Disc = Discriminator().to(device)
pretrained_Disc.load_state_dict(torch.load(pretrained_disciminator_path))
pretrained_Disc.eval()

# load the Pixel Valuation network
pretrained_pvrl_path = './models/lp_pretrained/netPv_rl_epoch_20.pth'
pretrained_PV = PVRL().to(device)
pretrained_PV.load_state_dict(torch.load(pretrained_pvrl_path))
pretrained_PV.eval()

# test adversarial examples in CIFAR10 training dataset
cifar_dataset = torchvision.datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    probs = pretrained_PV(test_img)
    probs_cpu = probs.cpu().detach()
    probs_np = probs_cpu.numpy()
    mask = np.random.binomial(1, probs_np, probs_np.shape)
    mask = torch.from_numpy(mask)
    mask = mask.view(mask.size(0), 1, test_img.size(2), test_img.size(3))
    mask = mask.type(torch.FloatTensor).to(device)
    test_img_with_mask = torch.cat((test_img, mask), 1).to(device)
    perturbation = pretrained_G(test_img_with_mask)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation * mask + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(targeted_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('CIFAR10 training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(cifar_dataset)))

# test adversarial examples in CIFAR10 testing dataset
cifar_dataset_test = torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(cifar_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    probs = pretrained_PV(test_img)
    probs_cpu = probs.cpu().detach()
    probs_np = probs_cpu.numpy()
    mask = np.random.binomial(1, probs_np, probs_np.shape)
    mask = torch.from_numpy(mask)
    mask = mask.view(mask.size(0), 1, test_img.size(2), test_img.size(3))
    mask = mask.type(torch.FloatTensor).to(device)
    test_img_with_mask = torch.cat((test_img, mask), 1).to(device)
    perturbation = pretrained_G(test_img_with_mask)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation * mask + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(targeted_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(cifar_dataset_test)))

