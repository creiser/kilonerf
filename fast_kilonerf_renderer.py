import torch
import kilonerf_cuda
from utils import PerfMonitor, ConfigManager
from run_nerf_helpers import get_rays, replace_transparency_by_background_color

class FastKiloNeRFRenderer():
    def __init__(self, c2w, intrinsics, background_color, occupancy_grid, multi_network, domain_mins, domain_maxs, 
        white_bkgd, max_depth_index, min_distance, max_distance, performance_monitoring, occupancy_resolution, max_samples_per_ray, transmittance_threshold):
        
        self.set_camera_pose(c2w)
        self.intrinsics = intrinsics
        self.background_color = background_color
        self.occupancy_grid = occupancy_grid
        self.multi_network = multi_network
        self.domain_mins = domain_mins
        self.domain_maxs = domain_maxs
        self.white_bkgd = white_bkgd
        self.max_depth_index = max_depth_index
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.performance_monitoring = performance_monitoring 
        self.occupancy_resolution = occupancy_resolution 
        self.max_samples_per_ray = max_samples_per_ray 
        self.transmittance_threshold = transmittance_threshold
        
        # Get ray directions for abitrary render pose
        # Precompute distances between sampling points, which vary along the pixel dimension
        _, rays_d = get_rays(intrinsics, self.c2w) # H x W x 3
        direction_norms = torch.norm(rays_d, dim=-1) # H x W
        self.distance_between_samples = (1 / (self.max_depth_index - 1)) * (self.max_distance - self.min_distance)
        self.constant_dists = (self.distance_between_samples  * direction_norms).view(-1).unsqueeze(1) # H * W x 1
        
        self.rgb_map = torch.empty([self.intrinsics.H, self.intrinsics.W, 3], dtype=torch.float, device=occupancy_grid.device)
        self.rgb_map_pointer = self.rgb_map.data_ptr()
    
    def set_rgb_map_pointer(self, rgb_map_pointer):
        self.rgb_map = None
        self.rgb_map_pointer = rgb_map_pointer
    
    def set_camera_pose(self, c2w):
        self.c2w = c2w[:3, :4]

    def render(self):
        PerfMonitor.add('start')
        PerfMonitor.is_active = self.performance_monitoring
    
        rays_o, rays_d = get_rays(self.intrinsics, self.c2w, expand_origin=False)
        PerfMonitor.add('ray directions', ['preprocessing'])
        
        origin = rays_o
        directions = rays_d.reshape(-1, 3) # directions are *not* normalized.
        res = self.occupancy_resolution
        global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(directions.device)
        global_domain_size = global_domain_max - global_domain_min
        occupancy_resolution = torch.tensor(res, dtype=torch.long, device=directions.device)
        strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.int, device=directions.device) # assumes row major ordering
        voxel_size = global_domain_size / occupancy_resolution
        num_rays = directions.size(0)
        
        active_ray_mask = torch.empty(num_rays, dtype=torch.bool, device=directions.device)
        depth_indices = torch.empty(num_rays, dtype=torch.short, device=directions.device)
        acc_map = torch.empty([self.intrinsics.H, self.intrinsics.W], dtype=torch.float, device=directions.device)
        # the final transmittance of a pass will be the initial transmittance of the next
        transmittance = torch.empty([self.intrinsics.H, self.intrinsics.W], dtype=torch.float, device=directions.device)
        
        PerfMonitor.add('prep', ['preprocessing'])
        
        is_initial_query = True
        is_final_pass = False

        pass_idx = 0
        integrate_num_blocks = 40
        integrate_num_threads = 512
        while not is_final_pass:
        
            if type(self.max_samples_per_ray) is list:
                # choose max samples per ray depending on the pass
                # in the later passes we can sample more per ray to avoid too much overhead from too many passes
                current_max_samples_per_ray = self.max_samples_per_ray[min(pass_idx, len(self.max_samples_per_ray) - 1)]
            else:
                # just use the same number of samples for all passes
                current_max_samples_per_ray = self.max_samples_per_ray

            # Compute query indices along the rays and determine assignment of query location to networks
            # Tunable CUDA hyperparameters
            kernel_max_num_blocks = 40
            kernel_max_num_threads = 512
            version = 0
            query_indices, assigned_networks = kilonerf_cuda.generate_query_indices_on_ray(origin, directions, self.occupancy_grid, active_ray_mask, depth_indices, voxel_size,
                global_domain_min, global_domain_max, strides, self.distance_between_samples, current_max_samples_per_ray, self.max_depth_index, self.min_distance, is_initial_query, 
                kernel_max_num_blocks, kernel_max_num_threads, version)

            PerfMonitor.add('sample query points', ['preprocessing'])
                
            with_explicit_mask = True
            query_indices = query_indices.view(-1)
            assigned_networks = assigned_networks.view(-1)
            if with_explicit_mask:
                active_samples_mask = assigned_networks != -1
                assigned_networks = assigned_networks[active_samples_mask]
            # when with_expclit_mask = False: Sort all points, including those with assigned_network == -1
            # Points with assigned_network == -1 will be placed in the beginning and can then be filtered by moving the start of the array (zero cost)
                
            #assigned_networks, reorder_indices = torch.sort(assigned_networks) # sorting via PyTorch is significantly slower
            #reorder_indices = torch.arange(assigned_networks.size(0), dtype=torch.int32, device=assigned_networks.device)
            #kilonerf_cuda.sort_by_key_int16_int32(assigned_networks, reorder_indices) # stable sort does not seem to be slower/faster
            reorder_indices = torch.arange(assigned_networks.size(0), dtype=torch.int64, device=assigned_networks.device)
            kilonerf_cuda.sort_by_key_int16_int64(assigned_networks, reorder_indices)
            PerfMonitor.add('sort', ['reorder and backorder'])
            
            # make sure that also batch sizes are given for networks which are queried 0 points
            contained_nets, batch_size_per_network_incomplete = torch.unique_consecutive(assigned_networks, return_counts=True)
            if not with_explicit_mask:
                num_removable_points = batch_size_per_network_incomplete[0]
                contained_nets = contained_nets[1:].to(torch.long)
                batch_size_per_network_incomplete = batch_size_per_network_incomplete[1:]
            else:
                contained_nets = contained_nets.to(torch.long)
            batch_size_per_network = torch.zeros(self.multi_network.num_networks, device=query_indices.device, dtype=torch.long)
            batch_size_per_network[contained_nets] = batch_size_per_network_incomplete
            ends = batch_size_per_network.cumsum(0).to(torch.int32)
            starts = ends - batch_size_per_network.to(torch.int32)
            PerfMonitor.add('batch_size_per_network', ['reorder and backorder'])
            

            # Remove all points which are assigned to no network (those points are in empty space or outside the global domain)
            if with_explicit_mask:
                query_indices = query_indices[active_samples_mask]
            else:
                reorder_indices = reorder_indices[num_removable_points:] # just moving a pointer
                PerfMonitor.add('remove points', ['reorder and backorder'])
            
            # Reorder the query indices
            query_indices = query_indices[reorder_indices]
            #query_indices = kilonerf_cuda.gather_int32(reorder_indices, query_indices)
            query_indices = query_indices
            PerfMonitor.add('reorder', ['reorder and backorder'])
            
            num_points_to_process = query_indices.size(0) if query_indices.ndim > 0 else 0
            #print("#points to process:", num_points_to_process, flush=True)
            if num_points_to_process == 0:
                break
                    
            # Evaluate the network
            network_eval_num_blocks = -1 # ignored currently
            compute_capability = torch.cuda.get_device_capability(query_indices.device)
            if compute_capability == (7, 5):
                network_eval_num_threads = 512 # for some reason the compiler uses more than 96 registers for this CC, so we cannot launch 640 threads
            else:
                network_eval_num_threads = 640
            version = 0
            raw_outputs = kilonerf_cuda.network_eval_query_index(query_indices, self.multi_network.serialized_params, self.domain_mins, self.domain_maxs, starts, ends, origin,
                self.c2w[:3, :3].contiguous(), self.multi_network.num_networks, self.multi_network.hidden_layer_size,
                self.intrinsics.H, self.intrinsics.W, self.intrinsics.cx, self.intrinsics.cy, self.intrinsics.fx, self.intrinsics.fy, self.max_depth_index, self.min_distance, self.distance_between_samples,
                network_eval_num_blocks, network_eval_num_threads, version)
            PerfMonitor.add('fused network eval', ['network query'])

            # Backorder outputs 
            if with_explicit_mask:
                raw_outputs_backordered = torch.empty_like(raw_outputs)
                raw_outputs_backordered[reorder_indices] = raw_outputs
                #raw_outputs_backordered = kilonerf_cuda.scatter_int32_float4(reorder_indices, raw_outputs)
                del raw_outputs
                raw_outputs_full = torch.zeros(num_rays * current_max_samples_per_ray, 4, dtype=torch.float, device=raw_outputs_backordered.device)
                raw_outputs_full[active_samples_mask] = raw_outputs_backordered
            else:
                raw_outputs_full = torch.zeros(num_rays * current_max_samples_per_ray, 4, dtype=torch.float, device=raw_outputs.device)
                raw_outputs_full[reorder_indices] = raw_outputs
            PerfMonitor.add('backorder', ['reorder and backorder'])
            
            # Integrate sampled densities and colors along each ray to render the final image
            version = 0
            kilonerf_cuda.integrate(raw_outputs_full, self.constant_dists, self.rgb_map_pointer, acc_map, transmittance, active_ray_mask, num_rays, current_max_samples_per_ray,
                self.transmittance_threshold, is_initial_query, integrate_num_blocks, integrate_num_threads, version)
            is_final_pass = not active_ray_mask.any().item()
            
            is_initial_query = False
            if not is_final_pass:
                PerfMonitor.add('integration', ['integration'])
            pass_idx += 1
        
        if self.white_bkgd:
            kilonerf_cuda.replace_transparency_by_background_color(self.rgb_map_pointer, acc_map, self.background_color, integrate_num_blocks, integrate_num_threads)
        
        PerfMonitor.is_active = True
        PerfMonitor.add('integration', ['integration'])
        elapsed_time = PerfMonitor.log_and_reset(self.performance_monitoring)
        self.rgb_map = self.rgb_map.view(self.intrinsics.H, self.intrinsics.W, 3) if self.rgb_map is not None else None
        return self.rgb_map, elapsed_time
