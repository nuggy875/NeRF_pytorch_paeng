
import os
import sys
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from load_dataset import load_blender, load_custom


def show_3d(data_root: str, data_name: str):
    img, pose, hwf, i_split = load_blender(
        data_root, data_name, False)

    test_img = img[0]
    test_pose = pose[0]

    camera_angle_x = 0.6911112070083618

    test_point = [[0], [0], [-1], [1]]
    test_point = np.array(test_point)

    xs = []
    ys = []
    zs = []

    for p in pose:
        a = np.dot(p, test_point)
        xs.append(a[0][0])
        ys.append(a[1][0])
        zs.append(a[2][0])

    xs_np = np.array(xs)
    ys_np = np.array(ys)
    zs_np = np.array(zs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(30, 60)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.scatter(xs_np, ys_np, zs_np, marker='o', s=30)

    # fig, axs = plt.subplots(ncols=1, figsize=(10,10), subplot_kw={"projection":"3d"})

    plt.title('Data 3D Visualize')
    plt.savefig('./test_img/data_dist/axis_z-1.png')
    plt.show()

    return 0


if __name__ == "__main__":

    show_3d("/home/root/data/nerf_synthetic/lego", "blender")
