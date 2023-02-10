import torch
import os
import imageio
import numpy as np
from tqdm import tqdm

from config import LOG_DIR, get_args_parser
from dataset import load_blender, load_custom, load_llff
from dataset.render_pose import get_render_pose
from model.NeRF import NeRF
from model.PositionalEncoding import get_positional_encoder
from nerf_process import batchify_rays_and_render_by_chunk
from rays import make_o_d
from utils import getLPIPS, getSSIM, img2mse, mse2psnr, to8b


def test(idx, i_test, posenc, model, test_imgs, gt_intrinsic, gt_extrinsic, hw, opts):
    print(f'>>> Start Testing for idx : {idx}')
    model.eval()
    checkpoint = torch.load(os.path.join(LOG_DIR, opts.exp_name, opts.exp_name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_test_dir = os.path.join(LOG_DIR, opts.exp_name, opts.exp_name+'_{}'.format(idx), 'test_result')
    os.makedirs(save_test_dir, exist_ok=True)

    img_h, img_w = hw

    param_intrinsic = gt_intrinsic
    param_extrinsic = gt_extrinsic

    losses = []
    perform_PSNR = []
    perform_SSIM = []
    perform_LPIPS = []
    result_best = {'i': 0, 'loss': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}
    with torch.no_grad():
        for i, test_pose in enumerate(tqdm(param_extrinsic)):
            rays_o, rays_d = make_o_d(img_w, img_h, param_intrinsic, test_pose[:3, :4])

            pred_rgb_c, pred_disp_c, pred_rgb_f, pred_disp_f = batchify_rays_and_render_by_chunk(rays_o, rays_d, model, posenc, img_h, img_w, param_intrinsic, opts)

            if opts.N_samples_f == 0:
                pred_rgb = pred_rgb_c
                pred_disp = pred_disp_c
            else:
                pred_rgb = pred_rgb_f
                pred_disp = pred_disp_f

            # SAVE test image
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            rgb_np = rgb.cpu().numpy()
            disp = torch.reshape(pred_disp, [img_h, img_w, 1])
            disp_np = disp.cpu().numpy()

            rgb8 = to8b(rgb_np)
            disp8 = to8b(disp_np/np.nanmax(disp_np))

            savefilename = os.path.join(save_test_dir, '{:03d}.png'.format(i))
            imageio.imwrite(savefilename, rgb8)
            savefilename = os.path.join(save_test_dir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(savefilename, disp8)

            # GET loss & psnr
            target_img_flat = torch.reshape(test_imgs[i], [-1, 3])
            # >> GET PSNR
            img_loss = img2mse(pred_rgb, target_img_flat)
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # GET SSIM & LPIPS
            loss_ssim = getSSIM(pred=rgb, gt=test_imgs[i])
            loss_lpips = getLPIPS(pred=rgb, gt=test_imgs[i])

            losses.append(img_loss)
            perform_PSNR.append(psnr)
            perform_SSIM.append(loss_ssim)
            perform_LPIPS.append(loss_lpips)
            print('idx:{} | Loss:{} | PSNR:{} | SSIM:{} | LPIPS:{}'.format(
                i, img_loss, psnr, loss_ssim, loss_lpips))

            # save best result
            if result_best['psnr'] < psnr:
                result_best['i'] = i
                result_best['loss'] = loss
                result_best['psnr'] = psnr
                result_best['ssim'] = psnr
                result_best['lpips'] = psnr

    print('BEST Result for Testing) idx : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr'], result_best['ssim'], result_best['lpips']))

    f = open(os.path.join(save_test_dir, "_result.txt"), 'w')
    result_sum = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    result_best = {'psnr': 0, 'ssim': 0, 'lpips': 1}
    for i in range(len(losses)):
        line = 'idx:{}\tloss:{}\tpsnr:{}\tssim:{}\tlpips:{}\n'.format(i, losses[i], perform_PSNR[i], perform_SSIM[i], perform_LPIPS[i])
        result_sum['psnr'] = result_sum['psnr'] + perform_PSNR[i]
        result_sum['ssim'] = result_sum['ssim'] + perform_SSIM[i]
        result_sum['lpips'] = result_sum['lpips'] + perform_LPIPS[i]
        result_best['psnr'] = perform_PSNR[i] if result_best['psnr'] < perform_PSNR[i] else result_best['psnr']
        result_best['ssim'] = perform_SSIM[i] if result_best['ssim'] < perform_SSIM[i] else result_best['ssim']
        result_best['lpips'] = perform_LPIPS[i] if result_best['lpips'] > perform_LPIPS[i] else result_best['lpips']
        f.write(line)
    f.write('\nBest Value ) PSNR : {}\tSSIM : {}\tLPIPS : {}\n'.format(
        result_best['psnr'], result_best['ssim'], result_best['lpips']))
    f.write('Mean Value ) PSNR : {}\tSSIM : {}\tLPIPS : {}'.format(
        result_sum['psnr']/len(losses), result_sum['ssim']/len(losses), result_sum['lpips']/len(losses)))
    f.close()


def render(idx, posenc, model, gt_intrinsic, render_pose, hw, opts):
    '''
    default ) n_angle : 40 / single_angle = -1
    if single_angle is not -1 , it would result single rendering image.
    '''
    print(f'>>> Start Rendering for idx : {idx}')
    print('>>> data type : '.format(opts.data_type))
    if opts.data_type == "blender" or opts.data_type == 'custom':
        render_pose = get_render_pose(
            n_angle=opts.n_angle,
            single_angle=opts.single_angle,
            phi=opts.phi,
            nf=opts.nf)
    
    render_poses = torch.Tensor(render_pose).to(torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))

    model.eval()
    checkpoint = torch.load(os.path.join(
        LOG_DIR, opts.exp_name, opts.exp_name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_render_dir = os.path.join(LOG_DIR, opts.exp_name, opts.exp_name+'_{}'.format(idx), 'render_result')
    os.makedirs(save_render_dir, exist_ok=True)

    img_h, img_w = hw
    param_intrinsic = gt_intrinsic

    rgbs = []
    disps = []
    with torch.no_grad():
        for i, render_pose in enumerate(tqdm(render_poses)):
            print('RENDERING... idx: {}'.format(i))
            rays_o, rays_d = make_o_d(img_w, img_h, param_intrinsic, render_pose[:3, :4])

            pred_rgb_c, pred_disp_c, pred_rgb_f, pred_disp_f = batchify_rays_and_render_by_chunk(rays_o, rays_d, model, posenc, img_h, img_w, param_intrinsic, opts)

            if opts.N_samples_f == 0:
                rgb = torch.reshape(pred_rgb_c, [img_h, img_w, 3])
                disp = torch.reshape(pred_disp_c, [img_h, img_w])
            else:
                rgb = torch.reshape(pred_rgb_f, [img_h, img_w, 3])
                disp = torch.reshape(pred_disp_f, [img_h, img_w])

            rgb_np = rgb.cpu().numpy()
            disp_np = disp.cpu().numpy()
            rgbs.append(rgb_np)
            disps.append(disp_np/np.nanmax(disp_np))
            print(f'{i}\t{np.nanmax(disp_np)}\t\t{disp_np.min()}')

            if not opts.single_angle == -1:
                imageio.imwrite(os.path.join(save_render_dir, '{}_{}_{}_rgb.png'.format(opts.single_angle, str(opts.phi), str(opts.nf))), to8b(rgb_np))
            imageio.imwrite(os.path.join(save_render_dir, f'{i}_rgb.png'), rgb_np)
            imageio.imwrite(os.path.join(save_render_dir, f'{i}_disp.png'), disp_np/np.nanmax(disp_np))

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

    if opts.single_angle == -1:
        if opts.render_type == 'mp4':
            imageio.mimwrite(os.path.join(save_render_dir, "_rgb.mp4"), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(save_render_dir, "_disp.mp4"), to8b(disps), fps=30, quality=8)
        if opts.render_type == 'gif':
            imageio.mimwrite(os.path.join(save_render_dir, "_rgb.gif"), to8b(rgbs), duration=0.04)
            imageio.mimwrite(os.path.join(save_render_dir, "_disp.gif"), to8b(disps), duration=0.04)
