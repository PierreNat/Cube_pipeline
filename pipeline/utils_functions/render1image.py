import torch
import os
import argparse
import numpy as np
import math as m
from math import pi
from numpy.random import uniform
import sys
import neural_renderer as nr
import tqdm

# ---------------------------------------------------------------------------------
# convert the coordinate system of rendered image to be the same as the blender object exportation
# creation of the 3x3 rotation matrix and the 1x3 translation vector
# ---------------------------------------------------------------------------------


def AxisBlend2Rend(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):

    alpha = alpha - pi/2

    Rx = np.array([[1, 0, 0],
                   [0, m.cos(alpha), -m.sin(alpha)],
                   [0, m.sin(alpha), m.cos(alpha)]])

    Ry = np.array([[m.cos(beta), 0, -m.sin(beta)],
                   [0, 1, 0],
                   [m.sin(beta), 0, m.cos(beta)]])

    Rz = np.array([[m.cos(gamma), m.sin(gamma), 0],
                   [-m.sin(gamma), m.cos(gamma), 0],
                   [0, 0, 1]])


# create the rotation object matrix

    Rzy = np.matmul(Rz, Ry)
    Rzyx = np.matmul(Rzy, Rx)

    # R = np.matmul(Rx, Ry)
    # R = np.matmul(R, Rz)

    t = np.array([tx, -ty, -tz])

    return t, Rzyx

# ---------------------------------------------------------------------------------
# random set translation and rotation parameter
# ---------------------------------------------------------------------------------

def get_Man_param_R_t():  # translation and rotation manual

    constraint_x = 2.5
    constraint_y = 2.5

    constraint_angle = m.pi

    x = 0
    y = 0
    z = -6

    alpha = np.radians(0)
    beta = np.radians(0)
    gamma = np.radians(0)

    return alpha, beta, gamma, x, y, z

def get_param_R_t():  # translation and rotation

    constraint_x = 2.5
    constraint_y = 2.5

    constraint_angle = m.pi

    x = round(uniform(-constraint_x, constraint_x), 1)
    y = round(uniform(-constraint_y, constraint_y), 1)
    z = round( uniform(-15, -5), 1)

    alpha = round(uniform(-constraint_angle,constraint_angle), 1)
    beta = round(uniform(-constraint_angle,constraint_angle), 1)
    gamma = round(uniform(-constraint_angle,constraint_angle), 1)

    return alpha, beta, gamma, x, y, z


def get_param_t():  # only translation

    constraint_x = 2.5
    constraint_y = 2.5


    # draw random value of R and t in a specific span
    x = round(uniform(-constraint_x, constraint_x), 1)
    y = round(uniform(-constraint_y, constraint_y), 1)
    z = round( uniform(-15, -5), 1)


    alpha = 0
    beta = 0
    gamma = 0

    return alpha, beta, gamma, x, y, z


def get_param_R():  # only rotation

    constraint_angle = m.pi/2

    # draw random value of R and t in a specific span
    x = 0
    y = 0
    z = -7 # need to see the entire cube (if set to 0 we are inside it)

    # alpha = round(uniform(0, constraint_angle), 1)
    # beta = round(uniform(0, constraint_angle), 1)
    # gamma = round(uniform(0, constraint_angle), 1)

    alpha = round(uniform(-constraint_angle, constraint_angle), 1)
    beta = 0
    gamma = 0

    return alpha, beta, gamma, x, y, z


# append element ---------------------------------------
# in: a list of all element
# in: boolean to see if we are writing the first elelemnt, othewise append to the existing list


def appendElement(all_elem, elem, first):

    if first:
        all_elem = np.expand_dims(elem, 0)  # create first element array
        first = False
    else:
        elem_exp = np.expand_dims(elem, 0)
        all_elem = np.concatenate((all_elem , elem_exp)) #append element to existing array

    return all_elem, first



def BuildTransformationMatrix(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):

    # alpha = alpha - pi/2

    Rx = np.array([[1, 0, 0],
                   [0, m.cos(alpha), -m.sin(alpha)],
                   [0, m.sin(alpha), m.cos(alpha)]])

    Ry = np.array([[m.cos(beta), 0, m.sin(beta)],
                   [0, 1, 0],
                   [-m.sin(beta), 0, m.cos(beta)]])

    Rz = np.array([[m.cos(gamma), -m.sin(gamma), 0],
                   [m.sin(gamma), m.cos(gamma), 0],
                   [0, 0, 1]])


# create the rotation object matrix

    Rzy = np.matmul(Rz, Ry)
    Rzyx = np.matmul(Rzy, Rx)

    # R = np.matmul(Rx, Ry)
    # R = np.matmul(R, Rz)

    t = np.array([tx, ty, tz])

    return t, Rzyx

class camera_setttings():
    # instrinsic camera parameter are fixed

    resolutionX = 512  # in pixel
    resolutionY = 512
    scale = 1
    f = 35  # focal on lens
    sensor_width = 32  # in mm given in blender , camera sensor type
    pixels_in_u_per_mm = (resolutionX * scale) / sensor_width
    pixels_in_v_per_mm = (resolutionY * scale) / sensor_width
    pix_sizeX = 1 / pixels_in_u_per_mm
    pix_sizeY = 1 / pixels_in_v_per_mm

    Cam_centerX = resolutionX / 2
    Cam_centerY = resolutionY / 2

    K = np.array([[f/pix_sizeX,0,Cam_centerX],
                  [0,f/pix_sizeY,Cam_centerY],
                  [0,0,1]])  # shape of [nb_vertice, 3, 3]


    def __init__(self, R, t, vert): #R 1x3 array, t 1x2 array, number of vertices
        self.R =R
        self.t = t
        self.alpha = R[0]
        self.beta= R[1]
        self.gamma = R[2]
        self.tx = t[0]
        self.ty= t[1]
        self.tz=t[2]
        # angle in radian
        self.t_mat, self.R_mat = BuildTransformationMatrix(self.tx, self.ty, self.tz, self.alpha, self.beta, self.gamma)

        self.K_vertices = np.repeat(camera_setttings.K[np.newaxis, :, :], vert, axis=0)
        self.R_vertices = np.repeat(self.R_mat[np.newaxis, :, :], vert, axis=0)
        self.t_vertices = np.repeat(self.t_mat[np.newaxis, :], 1, axis=0)


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

    Rt = np.concatenate((R, t),
                        axis=None)  # create one array of parameter in radian, this arraz will be saved in .npy file

    cam = camera_setttings(R=R, t=t, vert=nb_vertices)  # degree angle will be converted  and stored in radian

    renderer = nr.Renderer(image_size=512, camera_mode='projection', dist_coeffs=None,
                           K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=1,
                           background_color=[255, 255, 255],
                           far=1000, orig_size=512,
                           light_intensity_ambient=1.0, light_intensity_directional=0, light_direction=[0, 1, 0],
                           light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])  # [1,1,1]
    # UNKNOWN: near? orig_size?

    images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
    image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0))  # float32 from 0 to 255

    return image

