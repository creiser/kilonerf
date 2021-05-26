import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.integrate as integrate
import math
from collections import deque
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools

from utils import *
from local_distill import create_multi_network_fourier_embedding, has_flag, create_multi_network
from multi_modules import build_multi_network_from_single_networks, query_multi_network
import kilonerf_cuda

# TODO: move this to utils.py
class Node:
    def __init__(self):
        pass

# This function actually builds an occupancy grid
def build_occupancy_tree(cfg, log_path):
    dev = torch.device('cuda')
    kilonerf_cuda.init_stream_pool(16) # TODO: cleanup
    kilonerf_cuda.init_magma()
    
    ConfigManager.init(cfg)
    
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(torch.device('cpu'))
    global_domain_size = global_domain_max - global_domain_min
    Logger.write('global_domain_min: {}, global_domain_max: {}'.format(global_domain_min, global_domain_max))
    
    pretrained_cfg = load_yaml_as_dict(cfg['pretrained_cfg_path'])
    if 'distilled_cfg_path' in pretrained_cfg:
        pretrained_cfg = load_yaml_as_dict(pretrained_cfg['distilled_cfg_path'])
        
    if 'discovery' in pretrained_cfg:
        for key in pretrained_cfg['discovery']:
            pretrained_cfg[key] = pretrained_cfg['discovery'][key]
    """
    else:
        # end2end from scratch case
        assert pretrained_cfg['model_type'] == 'multi_network', 'occupancy grid creation is only implemented for multi networks'
    """
    
    cp = torch.load(cfg['pretrained_checkpoint_path'])
    use_multi_network = pretrained_cfg['model_type'] == 'multi_network' or not ('model_type' in pretrained_cfg)
    if use_multi_network:
        position_num_input_channels, position_fourier_embedding = create_multi_network_fourier_embedding(1, pretrained_cfg['num_frequencies'])
        direction_num_input_channels, direction_fourier_embedding = create_multi_network_fourier_embedding(1, pretrained_cfg['num_frequencies_direction'])
        
        if 'model_state_dict' in cp:
            res = pretrained_cfg['fixed_resolution']
            network_resolution = torch.tensor(res, dtype=torch.long, device=torch.device('cpu'))
            num_networks = res[0] * res[1] * res[2]
            network_voxel_size = global_domain_size / network_resolution
        
            multi_network = create_multi_network(num_networks, position_num_input_channels, direction_num_input_channels, 4,
                'multimatmul_differentiable', pretrained_cfg).to(dev)
            multi_network.load_state_dict(cp['model_state_dict'])
             
            # Determine bounding boxes (domains) of all networks. Required for global to local coordinate conversion.
            domain_mins = []
            domain_maxs = []
            for coord in itertools.product(*[range(r) for r in res]):
                coord = torch.tensor(coord, device=torch.device('cpu'))
                domain_min = global_domain_min + network_voxel_size * coord
                domain_max = domain_min + network_voxel_size
                domain_mins.append(domain_min.tolist())
                domain_maxs.append(domain_max.tolist())
            domain_mins = torch.tensor(domain_mins, device=dev)
            domain_maxs = torch.tensor(domain_maxs, device=dev)
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
            linear_implementation = 'multimatmul_differentiable'
            multi_network = build_multi_network_from_single_networks(single_networks, linear_implementation=linear_implementation).to(dev)
            domain_mins = torch.tensor(domain_mins, device=dev)
            domain_maxs = torch.tensor(domain_maxs, device=dev)
    else:
        # Load teacher NeRF model:
        pretrained_nerf = load_pretrained_nerf_model(dev, cfg)
    

    occupancy_res = cfg['resolution']
    total_num_voxels = occupancy_res[0] * occupancy_res[1] * occupancy_res[2]
    occupancy_grid = torch.tensor(occupancy_res, device=dev, dtype=torch.bool)
    occupancy_resolution = torch.tensor(occupancy_res, dtype=torch.long, device=torch.device('cpu'))
    occupancy_voxel_size = global_domain_size / occupancy_resolution
    first_voxel_min = global_domain_min
    first_voxel_max = first_voxel_min + occupancy_voxel_size
    
    first_voxel_samples = []
    for dim in range(3):
        first_voxel_samples.append(torch.linspace(first_voxel_min[dim], first_voxel_max[dim], cfg['subsample_resolution'][dim]))
    first_voxel_samples = torch.stack(torch.meshgrid(*first_voxel_samples), dim=3).view(-1, 3)
    
    ranges = []
    for dim in range(3):
        ranges.append(torch.arange(0, occupancy_res[dim]))
    index_grid = torch.stack(torch.meshgrid(*ranges), dim=3)
    index_grid = (index_grid * occupancy_voxel_size).unsqueeze(3)
    
    points = first_voxel_samples.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(occupancy_res + list(first_voxel_samples.shape))
    points = points + index_grid
    points = points.view(total_num_voxels, -1, 3)
    num_samples_per_voxel = points.size(1)

    
    mock_directions = torch.empty(min(cfg['voxel_batch_size'], total_num_voxels), 3).to(dev)

    # We query in a fixed grid at a higher resolution than the occupancy grid resolution to detect fine structures.
    all_densities = torch.empty(total_num_voxels, num_samples_per_voxel)
    end = 0
    while end < total_num_voxels:
        print('sampling network: {}/{} ({:.4f}%)'.format(end, total_num_voxels, 100 * end / total_num_voxels))
        start = end
        end =  min(start + cfg['voxel_batch_size'], total_num_voxels)
        actual_batch_size = end - start
        points_subset = points[start:end].to(dev).contiguous() # voxel_batch_size x num_samples_per_voxel x 3
        mock_directions_subset = mock_directions[:actual_batch_size]
        density_dim = 3
        with torch.no_grad():
            if use_multi_network:
                result = query_multi_network(multi_network, domain_mins, domain_maxs, points_subset, mock_directions_subset,
                    position_fourier_embedding, direction_fourier_embedding, None, None, False, None, pretrained_cfg)[:, :, density_dim]
            else:
                mock_directions_subset = mock_directions_subset.unsqueeze(1).expand(points_subset.size())
                points_and_dirs = torch.cat([points_subset.reshape(-1, 3), mock_directions_subset.reshape(-1, 3)], dim=-1)
                result = pretrained_nerf(points_and_dirs)[:, density_dim].view(actual_batch_size, -1)
            all_densities[start:end] = result.cpu()
    del points, points_subset, mock_directions
    
    occupancy_grid = all_densities.to(dev) > cfg['threshold']
    del all_densities
    occupancy_grid = occupancy_grid.view(cfg['resolution'] + [-1])

    occupancy_grid = occupancy_grid.any(dim=3) # checks if any point in the voxel is above the threshold

    
    Logger.write('{} out of {} voxels are occupied. {:.2f}%'.format(occupancy_grid.sum().item(), occupancy_grid.numel(), 100 * occupancy_grid.sum().item() / occupancy_grid.numel()))
    
    occupancy_filename = log_path + '/occupancy.pth'
    torch.save(occupancy_grid, occupancy_filename)
    Logger.write('Saved occupancy grid to {}'.format(occupancy_filename))
    
def main():
    cfg, log_path = parse_args_and_init_logger()
    build_occupancy_tree(cfg, log_path)
    
if __name__ == '__main__':
	main()
	
