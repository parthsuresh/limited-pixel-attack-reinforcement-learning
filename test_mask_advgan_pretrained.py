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

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_pretrained_epoch_100.pth'
pretrained_G = Generator().to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in CIFAR10 training dataset
cifar_dataset = torchvision.datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    mask = torch.ones(test_img.size(0), 1, test_img.size(2), test_img.size(3))
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
    mask = torch.ones(test_img.size(0), 1, test_img.size(2), test_img.size(3))
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

# helper functions

plotloader = DataLoader(cifar_dataset_test, batch_size=5, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    output = output.cpu()
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    images = images.detach()
    images, labels = images.cpu(), labels.cpu()
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 4.8))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

plotiter = iter(plotloader)
test_img, test_label = plotiter.next()
test_img, test_label = test_img.to(device), test_label.to(device)
mask = torch.ones(test_img.size(0), 1, test_img.size(2), test_img.size(3))
mask = mask.type(torch.FloatTensor).to(device)
test_img_with_mask = torch.cat((test_img, mask), 1).to(device)
perturbation = pretrained_G(test_img_with_mask)
perturbation = torch.clamp(perturbation, -0.3, 0.3)
adv_img = perturbation * mask + test_img
adv_img = torch.clamp(adv_img, 0, 1)

fig = plot_classes_preds(targeted_model, adv_img, test_label)
plt.savefig('./images/pretrained_modified.png')

fig = plot_classes_preds(targeted_model, test_img, test_label)
plt.savefig('./images/original.png')
