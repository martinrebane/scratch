#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:25:51 2019

@author: Martin Rebane

"""
import torch.nn as nn

class PhoneNet(nn.Module):
  
  def __init__(self):
    super().__init__()
    # 2 convolutions
    self.conv1 = nn.Sequential(nn.Conv2d(3,6,3,1,1), nn.BatchNorm2d(6), nn.ReLU())
    self.conv2 = nn.Sequential(nn.Conv2d(6,12,3,1,1), nn.BatchNorm2d(12), nn.ReLU())
    
    # scale data W and H down
    self.pool1 = nn.MaxPool2d(3, 1)
    
    # keeps adding channels
    self.conv3 = nn.Sequential(nn.Conv2d(12,24,3,1,1), nn.BatchNorm2d(24), nn.ReLU())
    
  
  def forward(self, data):
    data = self.conv1(data)
    data = self.conv2(data)
    data = self.pool1(data)
    data = self.conv3(data)
    return data