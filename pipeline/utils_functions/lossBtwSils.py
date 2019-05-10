import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

def lossBtwSils(GD_Sils,ComputedSils, lossfunction):

    gt = GD_Sils #selection of ground truth segmentation
    cp = ComputedSils #selection of computed segmentation
    # plt.imshow(gt.cpu(), cmap='gray')
    # plt.show()
    # print(GD_Sils.max(), GD_Sils.min())
    # print(ComputedSils.max(), ComputedSils.min())
    input = (gt.cpu()).type(torch.FloatTensor)/255.
    target = (cp.cpu()).type(torch.FloatTensor)/255.
    # plt.imshow(gtfloat, cmap='gray')
    # plt.show()
    # print(gtfloat.max(), gtfloat.min())
    input.requires_grad = True
    # print(input.requires_grad)
    temp_loss = lossfunction(input.double(), target.double())
    return temp_loss


