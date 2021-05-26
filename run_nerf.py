import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import warnings
import itertools
from tqdm import tqdm, trange
from copy import copy
import math
import pathlib
from skimage.metrics import structural_similarity as calculate_ssim

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_nsvf_dataset import load_nsvf_dataset, CameraIntrinsics

import kilonerf_cuda
import yaml
from fast_kilonerf_renderer import FastKiloNeRFRenderer
from local_distill import create_multi_network_fourier_embedding, has_flag, create_multi_network
from multi_modules import build_multi_network_from_single_networks, extract_linears, query_multi_network
from utils import ConfigManager, PerfMonitor, parse_args_and_init_logger, Logger, create_nerf, get_random_directions, load_yaml_as_dict, LPIPS
from von_mises import sample_von_mises_3d

device = torch.device("cuda")
DEBUG = False
RESTART_EXIT_CODE = 3
FINISHED_EXIT_CODE = 0

# Changing working directory to script's directory so that script can be called from anywhere
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# TODO: move class out of local_distill.py (which gets loaded as __main__)
class Node:
    def __init__(self):
        pass

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, random_directions=None, background_color=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], random_directions=random_directions, background_color=background_color, **kwargs)
        if random_directions is not None:
            ret, mean_regularization_term = ret
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    if random_directions is None:
        return all_ret
    else:
        return all_ret, mean_regularization_term
        
def render(intrinsics, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., c2w_staticcam=None, use_random_directions=False,
                  random_direction_probability=-1, von_mises_kappa=-1, random_directions=None,
                  background_color=None,
                  **kwargs):
    cfg = kwargs['cfg']
    
    PerfMonitor.add('start')
    PerfMonitor.is_active = has_flag(cfg, 'performance_monitoring')
    
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(intrinsics, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # provide ray directions as input
    if not use_random_directions:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(intrinsics, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    else:
        viewdirs = torch.tensor(get_random_directions(rays_o.size(0)), dtype=torch.float, device=rays_o.device)
    
    if random_direction_probability > 0.:
        num_viewdirs = viewdirs.size(0)
        probs = torch.empty(num_viewdirs, dtype=torch.float, device=viewdirs.device)
        probs[:] = random_direction_probability
        mask = Bernoulli(probs).sample().bool()
        num_random_viewdirs = mask.sum()
        random_viewdirs = torch.tensor(get_random_directions(num_random_viewdirs), dtype=torch.float, device=viewdirs.device)
        viewdirs[mask] = random_viewdirs
    
    if von_mises_kappa > 0.:
        viewdirs = viewdirs.cpu().numpy()
        viewdirs = sample_von_mises_3d(viewdirs, von_mises_kappa)
        viewdirs = torch.tensor(viewdirs, dtype=torch.float, device=rays_o.device)
    
    PerfMonitor.add('ray directions', ['preprocessing'])
    
    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, random_directions=random_directions, background_color=background_color, **kwargs)
    if random_directions is not None:
        all_ret, mean_regularization_term = all_ret
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
    PerfMonitor.is_active = True
    PerfMonitor.add('integration', ['integration'])
    elapsed_time = PerfMonitor.log_and_reset(has_flag(cfg, 'performance_monitoring'))

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    all_ret = ret_list + [elapsed_time] + [ret_dict]
    if random_directions is None:
        return all_ret
    else:
        return all_ret, mean_regularization_term

def create_fast_kilonerf_renderer(dummy_c2w, intrinsics, render_kwargs):
    cfg = render_kwargs['cfg']
    return FastKiloNeRFRenderer(dummy_c2w, intrinsics, render_kwargs['background_color'], render_kwargs['occupancy_grid'],
        render_kwargs['multi_network'], render_kwargs['domain_mins'], render_kwargs['domain_maxs'], 
        render_kwargs['white_bkgd'], render_kwargs['N_samples'], render_kwargs['near'], render_kwargs['far'],
        has_flag(cfg, 'performance_monitoring'), cfg['occupancy']['resolution'], cfg['render']['max_samples_per_ray'], cfg['render']['transmittance_threshold'])
        
def render_to_screen(intrinsics, render_kwargs):
    orbit_camera_cfg = render_kwargs['cfg']['orbit_camera']
    center = np.asarray(orbit_camera_cfg['center'])
    orbit_camera = OrbitCamera(center, orbit_camera_cfg['radius'], orbit_camera_cfg['inclination'], orbit_camera_cfg['azimuth'], device)
    fast_kilonerf_renderer = create_fast_kilonerf_renderer(orbit_camera.c2w, intrinsics, render_kwargs)
    kilonerf_cuda.render_to_screen(fast_kilonerf_renderer, orbit_camera, intrinsics.W, intrinsics.H)

def render_path(render_poses, intrinsics, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    intrinsics = copy(intrinsics)
    if render_factor != 0:
        intrinsics.H = intrinsics.H // render_factor
        intrinsics.W = intrinsics.W // render_factor
        intrinsics.fx = intrinsics.fx / render_factor
        intrinsics.fy = intrinsics.fy / render_factor
        intrinsics.cx = intrinsics.cx / render_factor
        intrinsics.cy = intrinsics.cy / render_factor

    rgbs = []
    disps = []
    
    c2ws = [x[:3, :4] for x in render_poses]
    
    use_fast_sampling = 'cfg' in render_kwargs and render_kwargs['cfg'] is not None and\
        'render' in render_kwargs['cfg'] and has_flag(render_kwargs['cfg']['render'], 'fast_sampling')
    
    if use_fast_sampling:
        fast_kilonerf_renderer = create_fast_kilonerf_renderer(c2ws[0], intrinsics, render_kwargs)

    mse_list, psnr_list, ssim_list, lpips_list, elapsed_time_list = [], [], [], [], []
    
    calculate_quality_metrics = gt_imgs is not None and render_factor == 0

    for i, c2w in enumerate(tqdm(c2ws)):
        log_str = 'Rendered image: {:2d}/{} '.format(i + 1, len(render_poses))

        if use_fast_sampling:
            fast_kilonerf_renderer.set_camera_pose(c2w)
            rgb, elapsed_time = fast_kilonerf_renderer.render()
        else:
            rgb, disp, acc, elapsed_time = render(intrinsics, chunk=chunk, c2w=c2w, **render_kwargs)[:4]
            disps.append(disp.cpu().numpy())
        rgbs.append(rgb.cpu().numpy())

        if calculate_quality_metrics:
            num_nans = torch.isnan(rgb).sum().item()
            if num_nans > 0:
                print('WARNING: Rendered image contains {} nan values. Converting nans to (1., 0., 0.)'.format(num_nans))
                rgb[torch.isnan(rgb)] = torch.tensor([1., 0., 0.], device=device)
        
            if type(gt_imgs[i]) is np.ndarray:
                gt_img_pytorch = torch.tensor(gt_imgs[i], device=device)
                gt_img_numpy = gt_imgs[i]
            else:
                gt_img_pytorch = gt_imgs[i].to(device)
                gt_img_numpy = gt_imgs[i].cpu().numpy()
            mse = img2mse(rgb, gt_img_pytorch)
            psnr = mse2psnr(mse)
            ssim = calculate_ssim(rgb.cpu().numpy(), gt_img_numpy, data_range=gt_img_numpy.max() - gt_img_numpy.min(), multichannel=True)
            lpips = LPIPS.calculate(rgb, gt_img_pytorch)
            log_str += 'MSE: {:.6f}, PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}, '.format(mse.item(), psnr.item(), ssim, lpips.item())
            mse_list.append(mse.item())
            psnr_list.append(psnr.item())
            ssim_list.append(ssim)
            lpips_list.append(lpips.item())
        log_str += 'time: {:7.2f} ms'.format(elapsed_time * 1000)
        elapsed_time_list.append(elapsed_time)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            
        Logger.write(log_str)
    
    average_mse = average_psnr = average_ssim = average_lpips = 0
    if calculate_quality_metrics:
        average_mse = sum(mse_list) / len(mse_list)
        average_psnr = sum(psnr_list) / len(psnr_list)
        average_ssim = sum(ssim_list) / len(ssim_list)
        average_lpips = sum(lpips_list) / len(lpips_list)
    average_elapsed_time = sum(elapsed_time_list) / len(elapsed_time_list)
    Logger.write('average over image batch. MSE: {:.6f}, PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}, elapsed time: {:7.2f} ms'.format(
        average_mse, average_psnr, average_ssim, average_lpips, average_elapsed_time * 1000))
    if savedir is not None:
        numerical_results = {'mse': average_mse, 'psnr': average_psnr, 'ssim': average_ssim, 'lpips': average_lpips, 'elapsed_time': average_elapsed_time}
        numerical_results_filename = os.path.join(savedir, 'numerical_results.json')
        with open(numerical_results_filename, 'w') as numerical_results_file:
            json.dump(numerical_results, numerical_results_file)

    rgbs = np.stack(rgbs, 0)
    if len(disps) > 0:
        disps = np.stack(disps, 0)

    return rgbs, disps
    
# only used during training: slower but with backprop support, no early ray termination, supports abitrarily spaced points 
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, background_color=None, pytest=False, no_color_sigmoid=False):

    def raw2alpha(raw, dists):
        return 1. - torch.exp(-F.relu(raw) * dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    if no_color_sigmoid:
        rgb = torch.min(torch.max(raw[...,:3], torch.zeros_like(raw[...,:3])), torch.ones_like(raw[...,:3]))  # [N_rays, N_samples, 3]
    else:
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

    # Initially the transmittance should equal 1 for all rays
    initial_transmittance = torch.ones((alpha.shape[0], 1))
    transmittance = torch.cumprod(torch.cat([initial_transmittance, 1. - alpha + 1e-10], -1), -1)
    
    transmittance = transmittance[:, :-1] # all columns but last column
    weights = alpha * transmittance
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    acc_map = torch.sum(weights, -1)

    # Optionally add a white (default) or background of another custom solid color.
    if white_bkgd:
        rgb_map = rgb_map + replace_transparency_by_background_color(acc_map, background_color)
  
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))    
    return rgb_map, disp_map, acc_map, weights, depth_map, alpha, transmittance


def render_rays(ray_batch,
                network_fn=None,
                network_query_fn=None,
                N_samples=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                visualize_hot_intervals=False,
                root_nodes=None,
                position_fourier_embedding=None,
                direction_fourier_embedding=None,
                cfg=None,
                multi_network=None,
                domain_mins=None,
                domain_maxs=None,
                occupancy_grid=None,
                debug_network_color_map=None,
                use_network_jittering=False,
                random_directions=None,
                background_color=None):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])
    
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        
        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    
    def query_network():
        if network_query_fn is not None:
            return network_query_fn(pts, viewdirs, network_fn.model_coarse if N_importance > 0 else network_fn)
        else:
            return query_multi_network(multi_network, domain_mins, domain_maxs, pts, viewdirs,
                position_fourier_embedding, direction_fourier_embedding, occupancy_grid, debug_network_color_map, use_network_jittering, random_directions, cfg)
    
    raw = query_network()
    if random_directions is not None:
        raw, mean_regularization_term = raw
            
    if has_flag(cfg, 'visualize_global_domain'):
        epsilon = 0.001
        global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(device)
        inside_global_domain = torch.logical_and((pts > global_domain_min + epsilon).all(dim=2), (pts < global_domain_max - epsilon).all(dim=2))
        if has_flag(cfg, 'crop_to_global_domain'):
            raw[torch.logical_not(inside_global_domain)] = torch.tensor([0., 0., 0., 0.], device=device)
        else:
            raw[inside_global_domain] = torch.tensor([0., 0., 0., 10000], device=device)
    
    no_color_sigmoid = has_flag(cfg, 'no_color_sigmoid')
    rgb_map, disp_map, acc_map, weights, depth_map, alpha, transmittance = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, background_color,
        pytest=pytest, no_color_sigmoid=no_color_sigmoid)
        
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        raw = network_query_fn(pts, viewdirs, network_fn.model_fine)
        
        if 'custom_color_min' in cfg:
            pts_flat = pts.view(-1, 3)
            inside_box = torch.logical_and((torch.tensor(cfg['custom_color_min'], dtype=torch.float, device=device) <= pts_flat).all(dim=1),
                (pts_flat <= torch.tensor(cfg['custom_color_max'], dtype=torch.float, device=device)).all(dim=1))
            raw_flat = raw.view(-1, 4)
            raw_flat[inside_box, :3] = torch.tensor([1., 0., 0.], dtype=torch.float)
            raw = raw_flat.view(raw.size())
            del pts_flat, raw_flat

        rgb_map, disp_map, acc_map, weights, depth_map, alpha, transmittance = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, background_color, 
            pytest=pytest, no_color_sigmoid=no_color_sigmoid)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if visualize_hot_intervals:
        ret.update({'weights': weights, 'pts': pts})
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    if random_directions is None:
        return ret
    else:
        return ret, mean_regularization_term
    
def pretrain_constant(multi_network, position_fourier_embedding, direction_fourier_embedding, cfg):
    assert not has_flag(cfg, 'use_global_coordinates')
    batch_size = cfg['constant_pretraining']['batch_size']
    random_points = torch.rand((batch_size, 3), device=device, dtype=torch.float) * 2 - 1 # rand in [-1, 1]
    random_directions = torch.tensor(get_random_directions(batch_size), device=device, dtype=torch.float)
    batch_size_per_network = torch.tensor([batch_size] + [0 for _ in range(multi_network.num_networks - 1)], dtype=torch.long, device=device).cpu()
    
    # Positional encoding
    fourier_embedding_implementation = 'custom_kernel_v2' # pytorch
    if position_fourier_embedding is not None:
        embedded_points = position_fourier_embedding(random_directions.unsqueeze(0), implementation=fourier_embedding_implementation).squeeze(0)
    else:
        embedded_points = random_points
    del random_points
    if direction_fourier_embedding is not None:
        embedded_dirs = direction_fourier_embedding(random_directions.unsqueeze(0), implementation=fourier_embedding_implementation).squeeze(0)
    else:
        embedded_dirs = random_directions
    del random_directions
    embedded_points_and_dirs = [embedded_points, embedded_dirs]
    del embedded_points
    del embedded_dirs
    
    raw = multi_network(embedded_points_and_dirs, batch_size_per_network)
    target = torch.tensor(cfg['constant_pretraining']['target'], device=device)
    return F.mse_loss(raw, target.unsqueeze(0).expand(batch_size, 4))

def train(cfg, log_path, render_cfg_path):
    Logger.write('Using GPU: {}'.format(torch.cuda.get_device_name(0)))
    
    # "Render" config overwrites the config
    if render_cfg_path is not None:
        render_cfg = load_yaml_as_dict(render_cfg_path)
        for key in render_cfg:
            cfg[key] = render_cfg[key]

    if 'rng_seed' in cfg:
        np.random.seed(cfg['rng_seed'])
        torch.manual_seed(cfg['rng_seed'])
          
    # Copy config values from distillation phases to top level
    def copy_to_top_level(cfg):
        if 'final' in cfg:
            for key in cfg['final']:
                cfg[key] = cfg['final'][key]
        elif 'discovery' in cfg:
            for key in cfg['discovery']:
                cfg[key] = cfg['discovery'][key]
    
    finetuning_distilled = 'distilled_cfg_path' in cfg
    if finetuning_distilled:
        distilled_cfg = load_yaml_as_dict(cfg['distilled_cfg_path'])
        copy_to_top_level(distilled_cfg)

        # Add configs in distilled config to this config
        for key in cfg:
            distilled_cfg[key] = cfg[key]
        cfg = distilled_cfg
    else:
        copy_to_top_level(cfg)
        
    if has_flag(cfg, 'visualize_global_domain') and 'occupancy_cfg_path' in cfg:
        del cfg['occupancy_cfg_path']
        
    ConfigManager.init(cfg)

    # Load data
    background_color = None # white is default
    if cfg['dataset_type'] == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(cfg['dataset_dir'], cfg['llff_factor'],
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=cfg['llff_spherify'])
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, cfg['dataset_dir'])
        if not isinstance(i_test, list):
            i_test = [i_test]

        if cfg['llff_hold'] > 0:
            print('Auto LLFF holdout,', cfg['llff_hold'])
            i_test = np.arange(images.shape[0])[::cfg['llff_hold']]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if cfg['llff_no_ndc']:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif cfg['dataset_type'] == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(cfg['dataset_dir'], cfg['blender_half_res'], cfg['testskip'])
        print('Loaded blender', images.shape, render_poses.shape, hwf, cfg['dataset_dir'])
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.
    elif cfg['dataset_type'] == 'nsvf':
        test_traj_path = cfg['test_traj_path'] if 'test_traj_path' in cfg else None
        images, poses, intrinsics, near, far, background_color, render_poses, i_split = load_nsvf_dataset(cfg['dataset_dir'],  cfg['testskip'], test_traj_path)
        print('Loaded a NSVF-style dataset', images.shape, poses.shape, render_poses.shape, cfg['dataset_dir'])
        
        i_train, i_val, i_test = i_split
        if i_test.size == 0:
            i_test = i_val
        
        print(i_train.shape, i_val.shape, i_test.shape)
        
    elif cfg['dataset_type'] == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=cfg['deepvoxels_shape'],
                                                                 basedir=cfg['dataset_dir'],
                                                                 testskip=cfg['testskip'])

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, cfg['dataset_dir'])
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', cfg['dataset_type'], 'exiting')
        return
        
    if cfg['dataset_type'] == 'blender' or cfg['dataset_type'] == 'nsvf':
        if cfg['blender_white_background'] and images.shape[-1] == 4:
            print('Converting alpha to white.')
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
        
    if cfg['dataset_type'] != 'nsvf':
        H, W, focal = hwf
        intrinsics = CameraIntrinsics(int(H), int(W), focal, focal, W * .5, H * .5)
        del H, W, focal, hwf
        
    if background_color is not None:
        background_color = torch.tensor(background_color, dtype=torch.float, device=device)
    else:
        background_color = torch.ones(3, dtype=torch.float, device=device)
        
    if 'near' in cfg:
        near = cfg['near']
    if 'far' in cfg:
        far = cfg['far']
    
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max()
    Logger.write('global_domain_min: {}, global_domain_max: {}, near: {}, far: {}, background_color: {}'.format(global_domain_min, global_domain_max, near, far, background_color))
    
    render_subset = 'custom_path'
    if cfg['render_test']:
        render_subset = 'test'
    if 'render_subset' in cfg:
        render_subset = cfg['render_subset']

    if render_subset == 'train':
        i_render = i_train
    elif render_subset == 'val':
        i_render = i_val
    elif render_subset == 'test':
        i_render = i_test
    if 'render_subset_indices' in cfg:
        i_render = i_render[cfg['render_subset_indices']]
    if render_subset != 'custom_path':
        render_poses = np.array(poses[i_render]) 
            
    # Checkpoint loading
    checkpoint_filenames = [f for f in os.listdir(log_path) if 'checkpoint' in f]
    load_from_checkpoint = len(checkpoint_filenames) > 0
    load_from_distilled = False
    if load_from_checkpoint:
        checkpoint_path = log_path + '/' + sorted(checkpoint_filenames)[-1]
        Logger.write('Loading {}'.format(checkpoint_path))
        cp = torch.load(checkpoint_path)
        if 'root_nodes' in cp:
            load_from_distilled = True
    elif finetuning_distilled:
        cp = torch.load(cfg['distilled_checkpoint_path'])
        load_from_distilled = True
    else:
        Logger.write('No checkpoint found. Fresh start.')
     
    render_kwargs_train = {
        'perturb' : cfg['perturb'],
        'N_samples' : cfg['num_samples_per_ray'],
        'N_importance' : cfg['num_importance_samples_per_ray'],
        'white_bkgd' : cfg['blender_white_background'],
        'raw_noise_std' : cfg['raw_noise_std'],
        'near' : near,
        'far' : far,
        'random_direction_probability': cfg['random_direction_probability'],
        'von_mises_kappa': cfg['von_mises_kappa'],
        'background_color': background_color,
        'cfg': cfg # TODO: hacky to pass down the whole config
    }
    
    # Create model
    if cfg['model_type'] == 'single_network':
        model, embed_fn, embeddirs_fn = create_nerf(cfg)
        model = model.to(device)
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=cfg['network_chunk_size'])

        additional_kwargs = {
            'network_query_fn' : network_query_fn,
            'network_fn' : model
        }                                                            
        
    elif cfg['model_type'] == 'multi_network' or load_from_distilled:
        # Required for fast training
        kilonerf_cuda.init_stream_pool(16)
        kilonerf_cuda.init_magma()
        
        position_num_input_channels, position_fourier_embedding = create_multi_network_fourier_embedding(1, cfg['num_frequencies'])
        direction_num_input_channels, direction_fourier_embedding = create_multi_network_fourier_embedding(1, cfg['num_frequencies_direction'])
        
        root_nodes = occupancy_grid = None
        
        # End-to-end training
        if not load_from_distilled:
            res = cfg['fixed_resolution']
            network_resolution = torch.tensor(res, dtype=torch.long, device=torch.device('cpu'))
            num_networks = res[0] * res[1] * res[2]
            model = multi_network = create_multi_network(num_networks, position_num_input_channels, direction_num_input_channels, 4,
                'multimatmul_differentiable', cfg).to(device)
            
            global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(torch.device('cpu'))
            global_domain_size = global_domain_max - global_domain_min
            network_voxel_size = global_domain_size / network_resolution
            
            # Determine bounding boxes (domains) of all networks. Required for global to local coordinate conversion.
            domain_mins = []
            domain_maxs = []
            for coord in itertools.product(*[range(r) for r in res]):
                coord = torch.tensor(coord, device=torch.device('cpu'))
                domain_min = global_domain_min + network_voxel_size * coord
                domain_max = domain_min + network_voxel_size
                domain_mins.append(domain_min.tolist())
                domain_maxs.append(domain_max.tolist())
            domain_mins = torch.tensor(domain_mins, device=device)
            domain_maxs = torch.tensor(domain_maxs, device=device)
        # From distillation
        else:
            root_nodes = cp['root_nodes']

            # Merging individual networks into multi network for efficient inference
            single_networks = []
            domain_mins, domain_maxs = [], []
            nodes_to_process = root_nodes.copy()
            for node in nodes_to_process:
                if hasattr(node, 'network'):
                    node.network_index = len(single_networks)
                    single_networks.append(node.network)
                    domain_mins.append(node.domain_min)
                    domain_maxs.append(node.domain_max)
                else:
                    nodes_to_process.append(node.leq_child)
                    nodes_to_process.append(node.gt_child)
            linear_implementation = 'multimatmul_differentiable' if finetuning_distilled else 'multimatmul'
            
            model = multi_network = build_multi_network_from_single_networks(single_networks, linear_implementation=linear_implementation,
                view_dependent_dropout_probability=cfg['view_dependent_dropout_probability']).to(device)
            multi_network.activation = nn.ReLU(inplace=True) # TODO: make sure that other activation functions are also inplace
            domain_mins = torch.tensor(domain_mins, device=device)
            domain_maxs = torch.tensor(domain_maxs, device=device)
              
        if 'occupancy_cfg_path' in cfg:
            occupancy_cfg = load_yaml_as_dict(cfg['occupancy_cfg_path'])
            if  'occupancy'not in cfg:
                cfg['occupancy'] = {}
            for key in occupancy_cfg:
                cfg['occupancy'][key] = occupancy_cfg[key]
            Logger.write('Loading occupancy grid from {}'.format(cfg['occupancy_log_path']))
            occupancy_grid = torch.load(cfg['occupancy_log_path']).reshape(-1)
            
            if 'render' in cfg:
                global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(device)
                global_domain_size = global_domain_max - global_domain_min
                
                # Compute centers of occupancy voxels
                res = cfg['occupancy']['resolution']
                occupancy_resolution = torch.tensor(res, dtype=torch.long, device=device)
                occupancy_voxel_size = global_domain_size / occupancy_resolution
                occupancy_voxel_half_size = occupancy_voxel_size / 2
                occupancy_voxel_centers = []
                
                for dim in range(3):
                    occupancy_voxel_centers.append(torch.linspace(global_domain_min[dim] + occupancy_voxel_half_size[dim],
                                                             global_domain_max[dim] - occupancy_voxel_half_size[dim],
                                                             res[dim]))
                occupancy_voxel_centers = torch.stack(torch.meshgrid(*occupancy_voxel_centers), dim=3).view(-1, 3)
                
                # Use tight domain to avoid sampling white background unessecarily modeled inside NeRF (NSVF baseline should also have access to this)
                if 'tight_domain_min' in cfg['render'] and 'tight_domain_max' in cfg['render']:
                    print('[WARNING] Cropping domain to:', cfg['render']['tight_domain_min'], cfg['render']['tight_domain_max'])
                    occupancy_domain_mins = occupancy_voxel_centers - occupancy_voxel_half_size
                    occupancy_domain_maxs = occupancy_voxel_centers + occupancy_voxel_half_size
                    tight_domain_min = torch.tensor(cfg['render']['tight_domain_min'], device=device)
                    tight_domain_max = torch.tensor(cfg['render']['tight_domain_max'], device=device)
                    inside_tight_domain = torch.logical_and(torch.all(tight_domain_min < occupancy_domain_mins, dim=1), torch.all(occupancy_domain_maxs < tight_domain_max , dim=1))
                    occupancy_grid[torch.logical_not(inside_tight_domain)] = 0
                
                if has_flag(cfg['render'], 'fast_sampling'):
                    assigned_network_grid = torch.empty_like(occupancy_grid, dtype=torch.short, device=device)

                    # Write assignment of network to occupancy voxel
                    res = cfg['fixed_resolution']
                    network_resolution = torch.tensor(res, dtype=torch.long, device=device)
                    strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.long, device=device) # assumes row major ordering
                    network_voxel_size = global_domain_size / network_resolution
                    
                    assigned_network_grid = ((occupancy_voxel_centers - global_domain_min) / network_voxel_size).to(strides)
                    assigned_network_grid = (assigned_network_grid * strides).sum(dim=1).to(torch.short)

                    # Write -1 to empty regions
                    assigned_network_grid[occupancy_grid == 0] = -1
                    occupancy_grid = assigned_network_grid
                
                
        
        # For visualizing the spatial distribution of networks in space
        debug_network_color_map = None    
        if has_flag(cfg, 'render_debug_network_color_map'):
            #debug_network_color_map = torch.rand(multi_network.num_networks, 3)
            debug_network_color_map = torch.rand(res + [3])
            for x, y, z in itertools.product(*[range(i) for i in res]):
                color = [0.0, 1.0, 0.] if (x + ((y + z % 2) % 2)) % 2 else [0.0, 0.0, 1.0]
                debug_network_color_map[x, y, z] = torch.tensor(color, dtype=torch.float)
            debug_network_color_map = debug_network_color_map.view(-1, 3)

        additional_kwargs = {
            'root_nodes': root_nodes,
            'position_fourier_embedding': position_fourier_embedding,
            'direction_fourier_embedding': direction_fourier_embedding,
            'multi_network': multi_network,
            'domain_mins': domain_mins,
            'domain_maxs': domain_maxs,
            'occupancy_grid': occupancy_grid,
            'debug_network_color_map': debug_network_color_map
        }
    render_kwargs_train.update(additional_kwargs)
    
    # NDC only good for LLFF-style forward facing data
    if cfg['dataset_type'] != 'llff' or cfg['llff_no_ndc']:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = cfg['llff_lindisp']

    render_kwargs_test = render_kwargs_train.copy()
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['random_direction_probability'] = -1
    render_kwargs_test['von_mises_kappa'] = -1
    
    # Create optimizer
    cfg['initial_learning_rate'] = float(cfg['initial_learning_rate'])
    
    optimizer_type = cfg['optimizer_type'] if 'optimizer_type' in cfg else 'adam'
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['initial_learning_rate'], betas=(0.9, 0.999))
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=cfg['initial_learning_rate']) # no momentum

    start = 0
    if load_from_checkpoint and not load_from_distilled:
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        model.load_state_dict(cp['model_state_dict'])
        start = cp['global_step'] + 1
    global_step = start
    
    use_fused_network_eval_kernel = 'render' in cfg and has_flag(cfg['render'], 'fast_sampling')
    if use_fused_network_eval_kernel:
        multi_network.serialize_params()
    
    if has_flag(cfg, 'render_to_screen'):
        render_to_screen(intrinsics, render_kwargs_test)
        return
        
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    
    # Short circuit if only rendering out from trained model
    if cfg['render_only']:
        print('RENDER ONLY')
        model.eval()
        with torch.no_grad():
            if render_subset != 'custom_path':
                # load images for subset to be rendered (train/val/test)
                images = images[i_render]
            else:
                # For custom render path there are no ground truth images
                images = None
                
            testsavedir = log_path + '/render_' + render_subset
            if render_cfg_path is not None:
                render_cfg_name = pathlib.Path(render_cfg_path).stem
                testsavedir += '_' + render_cfg_name
                
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, intrinsics, cfg['chunk_size'], render_kwargs_test, gt_imgs=images,
                savedir=testsavedir, render_factor=cfg['render_factor'])
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = cfg['num_rays_per_batch']
    use_batching = not cfg['no_batching']
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(intrinsics, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    if cfg is not None and 'iterations' in cfg:
        N_iters = cfg['iterations'] + 1
    
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Load rng_state from checkpoints to avoid repeating sampling pattern (could be a big problem when checkpointing frequently)
    if has_flag(cfg, 'rng_seed_fix') and load_from_checkpoint and 'torch_rng_state' in cp:
        Logger.write('Loading rng state. torch: {}, torch cuda: {}, numpy: {}'.format(cp['torch_rng_state'].sum(), cp['torch_cuda_rng_state'].sum(), cp['numpy_rng_state'][1].sum()))
        torch.set_rng_state(cp['torch_rng_state'])
        torch.cuda.set_rng_state(cp['torch_cuda_rng_state'])
        np.random.set_state(cp['numpy_rng_state'])
    
    start = start + 1
    for i in trange(start, N_iters):
        model.train()
        time0 = time.time()
        
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(intrinsics, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < cfg['precrop_iterations']:
                    dH = int(intrinsics.H//2 * cfg['precrop_fraction'])
                    dW = int(intrinsics.W//2 * cfg['precrop_fraction'])
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(intrinsics.H//2 - dH, intrinsics.H//2 + dH - 1, 2*dH), 
                            torch.linspace(intrinsics.W//2 - dW, intrinsics.W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        Logger.write(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {cfg['precrop_iterations']}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, intrinsics.H-1, intrinsics.H), torch.linspace(0, intrinsics.W-1, intrinsics.W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        
        use_random_directions = False
        if 'random_directions_iterations' in cfg:
            use_random_directions = i < cfg['random_directions_iterations']
            
        if 'network_jitter' in cfg:
            render_kwargs_train['use_network_jittering'] = i < cfg['network_jitter']['iterations']
            
        do_constant_pretraining = False
        if 'constant_pretraining' in cfg:
            do_constant_pretraining = i < cfg['constant_pretraining']['iterations']
            
            # Copy weights from network 0 to all remaining networks
            if i == cfg['constant_pretraining']['iterations']:
                with torch.no_grad():
                    linears = extract_linears(multi_network)
                    for linear in linears:
                        linear.weight[1:] = linear.weight[0]
                        linear.bias[1:] = linear.bias[0]
        
        random_directions = None
        if 'mean_regularization' in cfg:
            random_directions = torch.tensor(get_random_directions(cfg['mean_regularization']['num_random_directions']), dtype=torch.float, device=device)
            random_directions = direction_fourier_embedding(random_directions.unsqueeze(0), implementation='custom_kernel_v2').squeeze()

        if do_constant_pretraining: # assuming local coordinates
            loss = pretrain_constant(multi_network, position_fourier_embedding, direction_fourier_embedding, cfg)
        else:
            #####  Core optimization loop  #####
            all_ret = render(intrinsics, chunk=cfg['chunk_size'], rays=batch_rays,
                                                    verbose=i < 10, retraw=True, use_random_directions=use_random_directions,
                                                    random_directions=random_directions, **render_kwargs_train)
            if random_directions is not None:
                all_ret, mean_regularization_term = all_ret                               
            rgb, disp, acc, elapsed_time, extras = all_ret

            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)
            
            if 'l2_regularization_lambda' in cfg:
                l2_reg_term = multi_network.view_dependent_parameters[0].norm(2)
                for param in multi_network.view_dependent_parameters[1:]:
                    l2_reg_term = l2_reg_term + param.norm(2)
                l2_loss = cfg['l2_regularization_lambda'] * l2_reg_term
                loss = loss + l2_loss
                
            if random_directions is not None:
                mean_regularization_loss = cfg['mean_regularization']['weight'] * mean_regularization_term
                loss = loss + mean_regularization_loss
            
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Decay learning rate
        if cfg['learing_rate_decay_rate']:
            decay_rate = 0.1
            decay_steps = cfg['learing_rate_decay_rate'] * 1000
            new_lrate = cfg['initial_learning_rate'] * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
                
        dt = time.time()-time0

        # Rest is logging
        if i % cfg['render_video_interval'] == 0 and i > 0:
            # Turn on testing mode
            model.eval()
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, intrinsics, cfg['chunk_size'], render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = log_path + '/spiral_{:06d}_'.format(i)
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % cfg['render_testset_interval'] == 0 and i > 0:
            testsavedir = log_path + '/testset_{:06d}'.format(i)
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            model.eval()
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), intrinsics, cfg['chunk_size'], render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            Logger.write('Saved test set')

        if i % cfg['print_interval'] == 0 and not do_constant_pretraining:
            log_str = f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}"
            if 'l2_regularization_lambda' in cfg:
                log_str += f' L2 reg: {l2_loss.item()}'
            if 'mean_regularization' in cfg:
                log_str += f' mean reg: {mean_regularization_loss.item()}'
            Logger.write(log_str)
            tqdm.write(log_str)

        if i % cfg['checkpoint_interval'] == 0:
            torch_rng_state = torch.get_rng_state()
            torch_cuda_rng_state = torch.cuda.get_rng_state()
            numpy_rng_state = np.random.get_state()
            if has_flag(cfg, 'rng_seed_fix'):
                Logger.write('Saving rng state. torch: {}, torch cuda: {}, numpy: {}'.format(torch_rng_state.sum(), torch_cuda_rng_state.sum(), numpy_rng_state[1].sum()))
            cp = {
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'torch_rng_state': torch_rng_state,
                'torch_cuda_rng_state': torch_cuda_rng_state,
                'numpy_rng_state': numpy_rng_state
            }
            checkpoint_path = log_path + '/checkpoint_{:07d}.pth'.format(i)
            torch.save(cp, checkpoint_path)
            Logger.write('Saved checkpoint at {}'.format(checkpoint_path))
            
            # Restart the job after each checkpoint, if we are running on the MPI cluster.
            running_on_mpi_cluster = os.getenv('MPI_CLUSTER') == '1'
            if running_on_mpi_cluster:
                Logger.write('Restarting job.')
                exit(RESTART_EXIT_CODE)
                
        global_step += 1

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor') # sneaky
    cfg, log_path, render_cfg_path = parse_args_and_init_logger('default.yaml', parse_render_cfg_path=True)
    restarting_job = train(cfg, log_path, render_cfg_path)
    exit(FINISHED_EXIT_CODE)
    
if __name__ == '__main__':
    main()
