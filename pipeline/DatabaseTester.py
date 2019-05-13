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

# choose the database to test:
file_name_extension = '10000_t_zOnly'

cubes = np.load('Npydatabase/cubes_{}.npy'.format(file_name_extension))
sils = np.load('Npydatabase/sils_{}.npy'.format(file_name_extension))
params = np.load('Npydatabase/params_{}.npy'.format(file_name_extension))

for i in range(0, 10):
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


# #loss test -------------------------------------------------------------------------
#     loss = nn.CrossEntropyLoss()
#     loss2 = nn.BCELoss()
#
#
#     # input = torch.randn(10, 4, requires_grad=True)  #torch float 32
#     input = (torch.from_numpy(np.array([[1, 1, 1, 1, 1 ,0 ,0, 0, 0, 0],
#                                         [1, 1, 1, 1, 1 ,0 ,0, 0, 0, 0]])))
#     input2 = (torch.from_numpy(np.array([[0 ,0, 0, 0, 0 ,1, 1, 1, 1, 1 ],
#                                         [0 ,0, 0, 0, 0, 1, 1, 1, 1, 1 ]])))
#     input = input.double()
#     input2 = input2.double()
#     # target = input
#     # target = torch.empty(10, dtype=torch.long).random_(5) #torch int 64
#     target = torch.from_numpy(np.array([1, 1])) #torch int 64
#     target = target.long()
#     # input = (torch.from_numpy(np.array([[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]))).type(torch.)
#     # target = torch.from_numpy(np.array([1,2,3,4,5,6,7,8])).CharTensor
#     output = loss2(input,  input2)
#     Sil_1d_1 = torch.from_numpy((np.reshape(sil, np.size(sil,0)*np.size(sil,1))/255)).double()#size, [batch = 1, class = 512*512]
#
#     Sil_1d_2 = torch.from_numpy(np.reshape(sil2, np.size(sil2,0)*np.size(sil2,1))).long() #size = 512*512 values
#     Sil_1d_1 = Sil_1d_1.to(device)
#     Sil_1d_2 = Sil_1d_2.to(device)
#
#     #output should be n value
#     output = loss(Sil_1d_1, Sil_1d_1) #target should be a 1d tensor with n values, any value
#     print('loss computed')


