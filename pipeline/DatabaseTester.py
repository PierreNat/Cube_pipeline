#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:23:31 2019

@author: pierrec
"""
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import math as m
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

file_name_extension = '10000_t'

cubes = np.load('Npydatabase/cubes_{}.npy'.format(file_name_extension))
sils = np.load('Npydatabase/sils_{}.npy'.format(file_name_extension))
params = np.load('Npydatabase/params_{}.npy'.format(file_name_extension))

for i in range(0,10):
    fig = plt.figure()
    img = cubes[i]
    sil = sils[i]
    sil2 = sils[i+1]
    param = params[i]
    print(param)

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    fig.add_subplot(1, 2, 2)
    plt.imshow(sil, cmap='gray')
    plt.show()
    plt.close(fig)

    loss = nn.CrossEntropyLoss()
    input = torch.randn(10, 5, requires_grad=True) #torch float 32
    target = torch.empty(10, dtype=torch.long).random_(5) #torch int 64
    input = torch.from_numpy(np.array([[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]))
    target = torch.from_numpy(np.array([1,2,3,4,5,6,7,8])).CharTensor
    output = loss(input, target)
    Sil_1d_1 = torch.from_numpy((np.reshape(sil, np.size(sil,0)*np.size(sil,1))/255)) #size, [batch = 1, class = 512*512]

    Sil_1d_2 = torch.from_numpy(np.reshape(sil2, np.size(sil2,0)*np.size(sil2,1))) #size = 512*512 values
    Sil_1d_1 = Sil_1d_1.to(device)
    Sil_1d_2 = Sil_1d_2.to(device)

    #output should be n value
    output = loss(Sil_1d_1, Sil_1d_1) #target should be a 1d tensor with n values, any value
    print('loss computed')

