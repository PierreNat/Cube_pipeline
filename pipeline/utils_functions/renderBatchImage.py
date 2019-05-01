import numpy as np
from pipeline.utils_functions.render1image import render_1_sil, render_1_image
import matplotlib.pyplot as plt


def renderBatchImage(Obj_Name, predicted_params, device):
    batch_silhouettes = []  # create a void list for the rendered silhouette
    nbrOfParam = np.shape(predicted_params)[0]

    for i in range(0, nbrOfParam):
        # define extrinsic parameter
        sil = render_1_sil(Obj_Name, predicted_params[i])
        plt.imshow(sil, cmap='gray')
        # plt.show()
        # plt.close()
        batch_silhouettes.extend(sil)

    sils_database = np.reshape(batch_silhouettes, (nbrOfParam, 512, 512))  # shape(6, 512, 512) ndarray
    return sils_database.to(device)

