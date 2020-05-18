import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from generator_mask import Generator
from discriminator import Discriminator
from lp_pretrained_attack_func import PVRL_Attack
import resnet_model

use_cuda=True
epochs = 100
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Load target model
checkpoint = torch.load("./save_temp/checkpoint.th")
targeted_model = torch.nn.DataParallel(resnet_model.__dict__['resnet32']())
targeted_model.cuda()
targeted_model.load_state_dict(checkpoint['state_dict'])
targeted_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/mask_adv_gan/netG_pretrained_epoch_100.pth'
pretrained_G = Generator().to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.train()

# load the discriminator of adversarial examples
pretrained_discriminator_path = './models/mask_adv_gan/netDisc_pretrained_epoch_100.pth'
pretrained_Disc = Discriminator().to(device)
pretrained_Disc.load_state_dict(torch.load(pretrained_discriminator_path))
pretrained_Disc.train()

num_classes = 10

# CIFAR train dataset and dataloader declaration
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
pvrl = PVRL_Attack(device=device, \
    model=targeted_model, \
    generator=pretrained_G, \
    discriminator=pretrained_Disc, \
    model_num_labels=num_classes,\
    box_min=BOX_MIN,\
    box_max=BOX_MAX)


pvrl.train(dataloader, epochs)