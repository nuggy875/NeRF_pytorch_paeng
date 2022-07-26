import torch
import numpy as np


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


# theta, -30, 4 : shperical to cartesian
# theta : [-180, -171, -162 ..., 171] (40)
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def get_render_pose(n_angle=1, single_angle=-1, phi=-30.0):
    if not n_angle == 1 and single_angle == -1:
        render_poses = torch.stack([pose_spherical(angle, phi, 4.0)
                                    for angle in np.linspace(-180, 180, n_angle+1)[:-1]], 0)
    else:
        render_poses = pose_spherical(single_angle, phi, 4.0).unsqueeze(0)
    return render_poses


if __name__ == "__main__":
    render_poses = get_render_pose(n_angle=1, single_angle=120, phi=-30)
    print(render_poses.shape)
