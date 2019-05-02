import numpy as np
import tqdm
import torch

def lossBtwSils(GD_Sils,CompSils, lossfunction):

    gt = GD_Sils[0]
    predicted = CompSils[0]

    temp_loss = lossfunction(gt, predicted)
    return temp_loss


