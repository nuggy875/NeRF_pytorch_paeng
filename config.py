import configargparse
import os

LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), "logs")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    # config
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', is_config_file=True, help='config file path')

    # == Visualization with Visdom
    parser.set_defaults(visdom=True)
    parser.add('--visdom_port', type=int, default=8900)

    # == GPU
    parser.add('--gpu_ids', nargs="+")

    # ====== Dataset
    parser.add('--data_type', type=str, help='[ blender, llff, custom ]')
    parser.add('--data_name', type=str)
    parser.add('--data_root', type=str)
    parser.add('--downsample', type=int, default=0, help='0 to set downsample False')
    parser.add('--near', type=float)
    parser.add('--far', type=float)

    # ====== For blender
    parser.set_defaults(bkg_white=False)
    parser.add_argument('--bkg_white_true', dest='bkg_white', action='store_true')

    # ====== For colmap
    parser.set_defaults(colmap_relaunch=False)
    parser.add_argument('--colmap_relaunch_true', dest='colmap_relaunch', action='store_true')

    # ====== For precrop
    parser.add('--precrop_iters', type=int, default=0)
    parser.add('--precrop_frac', type=int, default=.5)

    # ====== For custom
    parser.add('--video_batch', type=int)

    # ====== Model
    parser.add('--L_x', type=int, default=10)
    parser.add('--L_d', type=int, default=4)
    parser.add('--netDepth', type=int, default=8)
    parser.add('--netWidth', type=int, default=256)

    # ====== Training
    parser.add('--exp_name', type=str)
    parser.add('--lr', type=float, default=5e-4)
    parser.add('--lr_min', type=float, default=5e-5)
    parser.add('--iter_warmup', type=int, default=10000)
    parser.add('--iter_N', type=int, help="Training Iteration")
    parser.add('--iter_start', type=int, default=0)

    # ====== Batch
    parser.set_defaults(global_batch=True)
    parser.add_argument('--global_batch_false', dest='global_batch', action='store_false')

    parser.add('--N_rays', type=int, default=4096)
    parser.add('--N_samples_c', type=int, default=64)
    parser.add('--N_samples_f', type=int, default=128)
    parser.add('--chunk_rays', type=int, default=4096)
    parser.add('--chunk_pts', type=int, default=524288, help='65536, 524288')
    parser.add('--perturb', default=1.)

    # ====== Testing
    parser.set_defaults(mode_test=True)
    parser.add_argument('--mode_test_false', dest='mode_test', action='store_false')
    parser.add('--testskip', type=int)

    # ====== Rendering
    parser.set_defaults(mode_render=True)
    parser.add_argument('--mode_render_false', dest='mode_render', action='store_false')
    parser.add('--render_type', type=str, help='mp4, gif', default='gif')

    # ====== (for blender & custom)
    parser.add('--n_angle', type=int)
    parser.add('--single_angle', type=float, default=-1)
    parser.add('--phi', type=float)
    parser.add('--nf', type=float)

    # ====== for only testing & rendering
    parser.add('--testing_idx', type=int)

    # ====== Set Index
    parser.add('--idx_vis', type=int, default=100)
    parser.add('--idx_print', type=int, default=1000)
    parser.add('--idx_save', type=int)
    parser.add('--idx_test', type=int)
    parser.add('--idx_render', type=int)
    parser.add('--idx_vis_cam_param', type=int, default=1000)

    opts = parser.parse_args()
    opts.world_size = len(opts.gpu_ids)

    for idx, device_num in enumerate(opts.gpu_ids):
        opts.gpu_ids[idx] = int(device_num)

    return opts


if __name__ == '__main__':
    opts = get_args_parser()
    print(opts)
    print(LOG_DIR)
    