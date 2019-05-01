import numpy as np
from pipeline.utils_functions.render1image import render_1_sil
import matplotlib.pyplot as plt



def renderBatchImage(Obj_Name, predicted_params):
    batch_silhouettes = []  # create a void list for the rendered silhouette
    nbrOfParam = np.shape(predicted_params)[0]

    for i in range(0, nbrOfParam):
        # define extrinsic parameter
        sil = render_1_sil(Obj_Name, predicted_params[i])
        plt.imshow(sil, cmap='gray')
        plt.show()
        plt.close()
        batch_silhouettes.extend(sil)

    return batch_silhouettes

