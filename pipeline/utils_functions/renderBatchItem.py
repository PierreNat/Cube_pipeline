import numpy as np
from utils_functions.render1item import render_1_sil, render_1_image
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

import numpy as np
from torchvision.transforms import ToTensor


def renderBatchSil(Obj_Name, predicted_params, ground_Truth, loss_function, device, plot=False):
    batch_silhouettes = []  # create a void list for the rendered silhouette
    # ground_Truth[:][5] = ground_Truth[:][5]+ torch.randn(1).uniform_(0, 1)
    nbrOfParam = predicted_params.size()[0]
    nb_im = nbrOfParam
    loss = 0

    for i in range(0, nbrOfParam):
        # define extrinsic parameter
        # predicted_params[i] = np.array([0, 0, 0, 2, 0.5*i, 8]) #test to enforce defined parameter
        sil_cp = render_1_sil(Obj_Name, predicted_params[i])


        sil_GT = render_1_sil(Obj_Name, ground_Truth[i])
        # print(torch.max(sil_cp))
        # loss += torch.sum((sil_cp - sil_GT) ** 2)
        sil_cp2 = sil_cp.squeeze() #from [1,512,512] to [512,512]
        sil_GT2 = sil_GT.squeeze().detach()
        # print(ground_Truth[i][5])
        # ground_Truth[i][5] = ground_Truth[i][5] + torch.randn(1).uniform_(0,1)

        distanc_im = sil_GT.detach().cpu().numpy().transpose((1, 2, 0))
        # plt.imshow(np.squeeze(distanc_im))
        # plt.show()
        # new_dist_im = np.where(distanc_im == 0, -1, distanc_im)*-1
        # dist_image = skfmm.distance(new_dist_im)
        # dist_image = np.squeeze(dist_image)
        # # print(image.shape)
        # # plt.imshow(dist_image)
        # # plt.show()
        #
        # dist_image_tensor = torch.from_numpy(dist_image).to(device)
        # pred_dist = (sil_cp2.double()*dist_image_tensor)
        # loss += loss_function(sil_cp2, sil_GT2) + (nn.MSELoss()(pred_dist, sil_GT2.double()).float())
        # TODO try a different loss for cp and gt? binary
        loss += loss_function(sil_cp2, sil_GT2) + nn.MSELoss()(predicted_params[i], ground_Truth[i]+(torch.randn(6)/10).to(device)) #compute loss and add constrain and noise
        # loss += torch.sum((sil_cp - sil_GT) ** 2)

        # if we want to see the result
        if plot:
            sil_GT =  render_1_sil(Obj_Name, ground_Truth[i])
            # plt.subplot(1, nb_im, i + 1)
            # plt.imshow(sil, cmap='gray')

            #conversion to use numpy imshow
            sil_cp = sil_cp.detach().cpu().numpy().transpose((1, 2, 0))
            sil_cp = np.squeeze((sil_cp * 255)).astype(np.uint8)
            sil_GT = sil_GT .detach().cpu().numpy().transpose((1, 2, 0))
            sil_GT  = np.squeeze((sil_GT  * 255)).astype(np.uint8)

            plt.subplot(2, nb_im, i + 1)
            plt.imshow(sil_GT, cmap='gray')

            plt.subplot(2, nb_im, i + 1 + nb_im)
            plt.imshow(sil_cp, cmap='gray')



    # plt.show()
    #
    #
    # sils_database = np.reshape(batch_silhouettes, (nbrOfParam, 512, 512))  # shape(6, 512, 512) ndarray
    # sils_database = torch.from_numpy(sils_database)
    # return sils_database.to(device)

    plt.show()
    return loss/nbrOfParam


def renderBatchImage(Obj_Name, predicted_params, device):
    batch_images = []  # create a void list for the rendered silhouette
    nbrOfParam = np.shape(predicted_params)[0]

    for i in range(0, nbrOfParam):
        # define extrinsic parameter
        sil = render_1_image(Obj_Name, predicted_params[i])
        # plt.imshow(sil, cmap='gray')
        # plt.show()
        # plt.close()
        batch_images.extend(sil)

    images_database = np.reshape(batch_images, (nbrOfParam, 512, 512, 3))  # shape(6, 512, 512) ndarray
    return images_database

