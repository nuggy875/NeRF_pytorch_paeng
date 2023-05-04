import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from rays import make_o_d

from dataset.load_custom import load_custom
from dataset.load_llff import load_llff
from dataset.load_blender import load_blender

from config import get_args_parser, LOG_DIR

def make_z_vals(N_pts=20, hw=None, near=2, far=6, device=None):
    img_h, img_w = hw
    # N_rays = img_h * img_w
    N_rays = 1
    t_vals = torch.linspace(
        0., 1., steps=N_pts, device=device)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_pts])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower+(upper-lower)*t_rand
    return z_vals


def visualize_extrinsic(iter, poses, idx_list, opts, vis_extrinsic=True, intrinsics = None, hw=None, debug_mode=False, idx_point=None, plt_show=False):
    # make Extrinsic [3, 4] -> [4, 4]
    device =  opts.gpu_ids[0]
    poses = poses[idx_list, :, :]
    bottom = np.reshape([0,0,0,1.], [1,1,4])
    bottom = np.tile(bottom, [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    #############################################################

    if opts.data_type == 'blender':
        default_point = [[0], [0], [0], [1]]
    elif opts.data_type == 'llff':
        default_point = [[0], [0], [0], [1]]
    else:
        default_point = [[0], [0], [0], [1]]
    default_point = np.array(default_point)

    xs = []
    ys = []
    zs = []
    xs_test = []
    ys_test = []
    zs_test = []
    # for pose in poses[:50]:
    for i, pose in enumerate(poses):
        a = np.dot(pose, default_point)
        xs.append(a[0][0])
        ys.append(a[1][0])
        zs.append(a[2][0])
        # if i == 0 or i==25 or i==50 or i==5:
        if debug_mode and idx_point is not None and i == idx_point:
            xs_test.append(a[0][0])
            ys_test.append(a[1][0])
            zs_test.append(a[2][0])
        
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    zs_np = np.array(zs)
    if debug_mode:
        xs_np_test = np.array(xs_test)
        ys_np_test = np.array(ys_test)
        zs_np_test = np.array(zs_test)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 60)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.scatter(xs_np, ys_np, zs_np, marker='o', s=20)
    if debug_mode and idx_point is not None:
        ax.scatter(xs_np_test, ys_np_test, zs_np_test, marker='o', s=30, color="#FF0066")

    # ax.set_zlim(0,1)


    ### SAVE IMAGE
    save_path = os.path.join(LOG_DIR, opts.exp_name, '_ext_vis')
    save_path_1 = os.path.join(save_path, '_t')
    save_path_2 = os.path.join(save_path, '_R_t')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(save_path_1):
        os.makedirs(save_path_1, exist_ok=True)
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2, exist_ok=True)
    if debug_mode:
        if plt_show:
            ax.view_init(elev=30., azim=120)
            plt.show()
        else:
            if idx_point is not None:
                ax.view_init(elev=30., azim=120)
                plt.savefig(save_path_1+f'/{iter}_30_120_{idx_point}.png')
            else:
                plt.savefig(save_path_1+f'/{iter}_30_120.png')
                ax.view_init(elev=60., azim=120)
            
    else:
        plt.savefig(save_path_1+f'/{iter}.png')

    # >>> VISUALIZE Extrinsic
    if vis_extrinsic:
        vec_pts_N = 120
        vec_near = opts.near
        vec_far = opts.far
        img_h, img_w = hw
        for i, target_pose in enumerate(poses):
            rays_o, rays_d = make_o_d(img_w, img_h, intrinsics, torch.from_numpy(target_pose).float().to(device))
            z_vals = make_z_vals(vec_pts_N, hw, vec_near, vec_far, device)
            center_vector = rays_d[int(img_h/2)][int(img_w/2)]
            o_val = rays_o[0][0]
            ray_test = o_val + center_vector.unsqueeze(0) * z_vals.reshape(vec_pts_N,1)
            xs = []
            ys = []
            zs = []
            for idx, point in enumerate(ray_test):
                xs.append(point[0].item())
                ys.append(point[1].item())
                zs.append(point[2].item())
            xs_np=np.array(xs)
            ys_np=np.array(ys)
            zs_np=np.array(zs)
            ax.scatter(xs_np, ys_np, zs_np, marker='o', s=0.1, color="#FF0066")

    ### SAVE IMAGE
    if debug_mode:
        if plt_show:
            ax.view_init(elev=30., azim=120)
            plt.show()
        else:
            if idx_point is not None:
                ax.view_init(elev=30., azim=120)
                plt.savefig(save_path_2+f'/{iter}_30_120_{idx_point}.png')
            else:
                ax.view_init(elev=30., azim=120)
                plt.savefig(save_path_2+f'/{iter}_30_120.png')
    else:
        plt.savefig(save_path_2+f'/{iter}.png')


def visualize_ray(rays_o, rays_d, hw, device):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 60)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    # rays_o : [H, W, 3]
    # rays_d : [H, W, 3]
    img_h, img_w = hw
    vec_pts_N = 120
    near = 0.0
    far = 2.0
    z_vals = make_z_vals(vec_pts_N, hw, near, far, device)
    center_vector = rays_d[int(img_h/2)][int(img_w/2)]
    o_val = rays_o[0][0]
    ray_test = o_val + center_vector.unsqueeze(0) * z_vals.reshape(vec_pts_N,1)
    xs = []
    ys = []
    zs = []
    for idx, point in enumerate(ray_test):
        xs.append(point[0].item())
        ys.append(point[1].item())
        zs.append(point[2].item())
    xs_np=np.array(xs)
    ys_np=np.array(ys)
    zs_np=np.array(zs)
    ax.scatter(xs_np, ys_np, zs_np, marker='o', s=0.1, color="#FF0066")
    plt.savefig('./logs'+f'/test_1.png')




if __name__ == "__main__":
    opts = get_args_parser()

    if opts.data_type == "blender":
        images, gt_camera_param, hw, i_split = load_blender(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            bkg_white=opts.bkg_white
        )
        render_poses = None
    elif opts.data_type == 'llff':
        images, gt_camera_param, hw, i_split, render_poses = load_llff(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            colmap_relaunch=opts.colmap_relaunch
        )
    elif opts.data_type == 'custom':
        images, gt_camera_param, hw, i_split, nf = load_custom(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            video_batch=opts.video_batch,
            colmap_relaunch=opts.colmap_relaunch
        )
        render_poses = None
        opts.near, opts.far = nf
    i_train, i_val, i_test = i_split
    img_h, img_w = hw
    
    i_total_llff = i_train

    (gt_intrinsic, gt_extrinsic) = gt_camera_param
    print(f"\n\n>>> Image shape : {images.shape}")
    
    # 1 ) Visualize
    visualize_extrinsic(iter=0, poses=gt_extrinsic, idx_list=i_total_llff, opts=opts, intrinsics=gt_intrinsic, hw=hw, debug_mode=True, idx_point=None, plt_show=False)
    
    # 2 ) Visualize per Index
    # for i, pose in enumerate(gt_extrinsic):
    #     visualize_extrinsic(iter=0, poses=gt_extrinsic, idx_list=i_total_llff, opts=opts, intrinsics=gt_intrinsic, hw=hw, debug_mode=True, idx_point=i, plt_show=False)
    