import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mask_advgan_pretrain_functions import AdvGAN_Pretrain
import resnet_model

use_cuda=True
epochs = 100
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

checkpoint = torch.load("./save_temp/checkpoint.th")
targeted_model = torch.nn.DataParallel(resnet_model.__dict__['resnet32']())
targeted_model.cuda()
targeted_model.load_state_dict(checkpoint['state_dict'])
targeted_model.eval()
num_classes = 10

# CIFAR train dataset and dataloader declaration
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
advGAN = AdvGAN_Pretrain(device=device, \
    model=targeted_model, \
    model_num_labels=num_classes,\
    box_min=BOX_MIN,\
    box_max=BOX_MAX)


advGAN.train(dataloader, epochs)
