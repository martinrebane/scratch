#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:20:02 2019

@author: martin
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import ResNetUNet
from pytorch_unet import UNet
import glob
from PIL import Image
import torchvision.transforms.functional as TF

# path to training data
path = '/media/martin/Acer/teleplan/AAA-img/20190404/img/backs/'


# pre-trained ResNet model requires input to be normalised, see pytorch docs
normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])    

class PhoneDataset(Dataset):
  def __init__(self, path):
    self.files = glob.glob(path + "*.jpg")
  
  def __getitem__(self, idx):
    file = self.files[idx]
    mask_file = file.replace(".jpg", "_def1.png")
    # open image
    img = Image.open(file)
    mask = Image.open(mask_file)
    # crop
    img = TF.center_crop(img, (1900, 900))
    mask = TF.center_crop(mask, (1900, 900))
    # convert to tensor
    img, mask = TF.to_tensor(img), TF.to_tensor(mask)

    return img, mask
  
  def __len__(self):
    return len(self.files)

# load data from any folder
#data = datasets.ImageFolder('/media/martin/Acer/teleplan/AAA-img/20190404/img/',
#                            transform=transforms.Compose([
#                                transforms.CenterCrop(192),
#                                transforms.ToTensor()
#                                ]))

data_loader = DataLoader(PhoneDataset(path),
                              batch_size=2,
                              shuffle=True
                              )
# set device, use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device.type)

#model = ResNetUNet(7)
model = UNet(5)
# use current device
model.to(device)

# use stochastic gradient descent
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

#use learning rate decay
scheduler = StepLR(optimiser, step_size=10, gamma=0.0005)


for epoch in range(3):
  # get a batch of images + masks
  for img, mask in data_loader:
    print("success")
    print(img.shape)
    # reset gradients from previous step
    optimiser.zero_grad()
    # forward propagation
    pred = model.forward(img)
    
    # update model parameters
    optimiser.step()
  # update learning rate
  scheduler.step()
  print("Finished epoch " + epoch)
  
print("End of the training")