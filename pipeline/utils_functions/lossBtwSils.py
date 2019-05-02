import numpy as np
import tqdm
import torch

def lossBtwSils(GD_Sils,CompSils, lossfunction):

    gt = GD_Sils[0]
    gtfloat = gt.type(torch.DoubleTensor)/255
    # print(gtfloat.max(), gtfloat.min())
    temp_loss = lossfunction(gt, predicted)
    return temp_loss


