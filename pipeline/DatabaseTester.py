#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:23:31 2019

@author: pierrec
"""
import matplotlib.pyplot as plt
import numpy as np
import math as m
import torch

file_name_extension = '10quicktest'

cubes = np.load('Npydatabase/cubes_{}.npy'.format(file_name_extension))
sils = np.load('Npydatabase/sils_{}.npy'.format(file_name_extension))
params = np.load('Npydatabase/params_{}.npy'.format(file_name_extension))

for i in range(0,10):
    fig = plt.figure()
    img = cubes[i]
    sil = sils[i]
    param = params[i]
    print(param)

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    fig.add_subplot(1, 2, 2)
    plt.imshow(sil, cmap='gray')
    plt.show()
    plt.close(fig)

