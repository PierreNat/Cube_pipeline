import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

def lossBtwSils(GD_Sils,ComputedSils, lossfunction):

    gt = GD_Sils[0] #selection of ground truth segmentation
    cp = ComputedSils[0] #selection of computed segmentation
    # plt.imshow(gt.cpu(), cmap='gray')
    # plt.show()
    # print(GD_Sils.max(), GD_Sils.min())
    # print(ComputedSils.max(), ComputedSils.min())
    gtfloat = (gt.cpu()).type(torch.FloatTensor)/255.
    cpfloat = (cp.cpu()).type(torch.FloatTensor)/255.
    # plt.imshow(gtfloat, cmap='gray')
    # plt.show()
    # print(gtfloat.max(), gtfloat.min())
    temp_loss = lossfunction(cpfloat, gtfloat)
    return temp_loss


