import numpy as np
from pipeline.utils_functions.camera_settings import camera_setttings
import neural_renderer as nr


def render_1_image(Obj_Name, params):
    print("creation of a single image")

    vertices_1, faces_1, textures_1 = nr.load_obj("./3D_objects/{}.obj".format(Obj_Name), load_texture=True)#, texture_size=4)
    print(vertices_1.shape)
    print(faces_1.shape)
    vertices_1 = vertices_1[None, :, :]  # add dimension
    faces_1 = faces_1[None, :, :]  #add dimension
    textures_1 = textures_1[None, :, :]  #add dimension
    nb_vertices = vertices_1.shape[0]

    # define extrinsic parameter
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    x = params[3]
    y = params[4]
    z = params[5]

    R = np.array([alpha, beta, gamma])  # angle in degree
    t = np.array([x, y, z])  # translation in meter

    Rt = np.concatenate((R, t), axis=None)  # create one array of parameter in radian, this arraz will be saved in .npy file

    cam = camera_setttings(R=R, t=t, vert=nb_vertices)  # degree angle will be converted  and stored in radian

    renderer = nr.Renderer(image_size=512, camera_mode='projection', dist_coeffs=None,
                           K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=1,
                           background_color=[255, 255, 255],
                           far=1000, orig_size=512,
                           light_intensity_ambient=1.0, light_intensity_directional=0, light_direction=[0, 1, 0],
                           light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])  # [1,1,1]

    images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
    image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0))  # float32 from 0 to 255

    return image

