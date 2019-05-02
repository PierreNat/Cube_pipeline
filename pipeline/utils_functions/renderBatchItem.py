import numpy as np
from pipeline.utils_functions.render1item import render_1_sil, render_1_image
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


def renderBatchSil(Obj_Name, predicted_params, device):
    batch_silhouettes = []  # create a void list for the rendered silhouette
    nbrOfParam = np.shape(predicted_params)[0]

    for i in range(0, nbrOfParam):
        # define extrinsic parameter
        sil = render_1_sil(Obj_Name, predicted_params[i])
        # plt.imshow(sil, cmap='gray')
        # plt.show()
        # plt.close()
        batch_silhouettes.extend(sil)

    sils_database = np.reshape(batch_silhouettes, (nbrOfParam, 512, 512))  # shape(6, 512, 512) ndarray
    sils_database = torch.from_numpy(sils_database)
    return sils_database.to(device)


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

