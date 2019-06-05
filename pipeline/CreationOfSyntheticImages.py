
import numpy as np
import torch
from numpy.random import uniform
import neural_renderer as nr
import matplotlib.pyplot as plt
import tqdm
from utils_functions.camera_settings import camera_setttings

def main():

    cubes_database = []
    sils_database = []
    params_database = []
    im_nr = 1

    vertices_1, faces_1, textures_1 = nr.load_obj("3D_objects/rubik_color.obj", load_texture=True)#, texture_size=4)
    print(vertices_1.shape)
    print(faces_1.shape)
    vertices_1 = vertices_1[None, :, :]  # add dimension
    faces_1 = faces_1[None, :, :]  #add dimension
    textures_1 = textures_1[None, :, :]  #add dimension
    nb_vertices = vertices_1.shape[0]

    print(vertices_1.shape)
    print(faces_1.shape)

    file_name_extension = '_10'
    nb_im = 10
    #init and create renderer object
    R = np.array([np.radians(0), np.radians(0), np.radians(0)])  # angle in degree
    t = np.array([0, 0, 0])  # translation in meter
    cam = camera_setttings(R=R, t=t, vert=nb_vertices)
    renderer = nr.Renderer(image_size=512, camera_mode='projection', dist_coeffs=None,
                           K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=1, background_color=[1, 1, 1], #background is filled now with  value 0-1 instead of 0-255
                           # changed from 0-255 to 0-1
                           far=1000, orig_size=512,
                           light_intensity_ambient=1.0, light_intensity_directional=0, light_direction=[0, 1, 0],
                           light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])

    loop = tqdm.tqdm(range(0, nb_im))
    for i in loop:
        # define transfomration parameter randomly uniform
        alpha = uniform(0, 180)
        beta =  uniform(0, 180)
        gamma = uniform(0, 180)
        x = uniform(-2, 2)
        y = uniform(-2, 2)
        z = uniform(8, 14)
        R = np.array([np.radians(alpha), np.radians(beta), np.radians(gamma)])  # angle in degree
        t = np.array([x, y, z])  # translation in meter

        Rt = np.concatenate((R, t), axis=None).astype(np.float16)  # create one array of parameter in radian, this arraz will be saved in .npy file

        cam = camera_setttings(R=R, t=t, vert=nb_vertices) # degree angle will be converted  and stored in radian

        images_1 = renderer(vertices_1, faces_1, textures_1,
                            K=torch.cuda.FloatTensor(cam.K_vertices),
                            R=torch.cuda.FloatTensor(cam.R_vertices),
                            t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0-1

        image = (image*255).astype(np.uint8) #cast from float32 255.0 to 255 uint8

        sils_1 = renderer(vertices_1, faces_1, textures_1,
                          mode='silhouettes',
                          K=torch.cuda.FloatTensor(cam.K_vertices),
                          R=torch.cuda.FloatTensor(cam.R_vertices),
                          t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        sil = sils_1.detach().cpu().numpy().transpose((1, 2, 0))
        sil = np.squeeze((sil * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]

        #grow the list of cube, silhouette and parameters
        cubes_database.extend(image)
        sils_database.extend(sil)
        params_database.extend(Rt)

        im_nr = im_nr+1



        if(im_nr%1 == 0):
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(image)

            fig.add_subplot(1, 2, 2)
            plt.imshow(sil, cmap='gray')
            plt.show()
            plt.close(fig)

# save database
# reshape in the form (nbr of image, x dim, y dim, layers)
    cubes_database = np.reshape(cubes_database, (im_nr-1, 512, 512, 3)) # 3 channel rgb
    sils_database = np.reshape(sils_database, (im_nr-1, 512, 512)) #binary mask monochannel
    params_database = np.reshape(params_database,(im_nr-1, 6)) #array of 6 params
    np.save('Npydatabase/cubes_{}.npy'.format(file_name_extension), cubes_database)
    np.save('Npydatabase/sils_{}.npy'.format(file_name_extension), sils_database)
    np.save('Npydatabase/params_{}.npy'.format(file_name_extension), params_database)
    print('images saved')


if __name__ == '__main__':
    main()
