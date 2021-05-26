#include "generate_inputs.cuh"

#include "utils.cuh"

#include <float.h>
#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

__device__ __constant__ float c_c2w[9];

__global__ void get_rays_d_kernel_0(const int H, const int W, const float cx, const float cy, const float fx, const float fy, float *rays_d)
{
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx_x < W) {
        int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
        while (idx_y < H) {
            float in[3] = {((float)idx_x - cx) / fx, -((float)idx_y - cy) / fy, -1};
            float out[3] = {0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                #pragma unroll
                for (int i = 0; i < 3; i++) {
                    out[i] += in[j] * c_c2w[i * 3 + j];
                }
            }
        
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                rays_d[(idx_x + idx_y * W) * 3 + i] = out[i];
            }
            idx_y += gridDim.y * blockDim.y; 
        }
        idx_x += gridDim.x * blockDim.x;
    }
}

torch::Tensor get_rays_d(const int& H, const int& W, const float& cx, const float& cy,  const float& fx, const float& fy, const torch::Tensor& c2w_tensor, const int& root_num_blocks, const int& root_num_threads)
{
    torch::Tensor rays_d_tensor = torch::empty({H, W, 3}, c2w_tensor.options());
    float *rays_d = rays_d_tensor.data_ptr<float>();
    float *c2w = c2w_tensor.data_ptr<float>();
    cudaMemcpyToSymbol(c_c2w, c2w, 9 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    dim3 gridDim(root_num_blocks, root_num_blocks);
    dim3 blockDim(root_num_threads, root_num_threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    get_rays_d_kernel_0<<<gridDim, blockDim, 0, stream>>>(H, W, cx, cy, fx, fy, rays_d);
    gpuErrchk( cudaPeekAtLastError() );
    #ifdef DEBUG_SYNC
    gpuErrchk( cudaDeviceSynchronize() );
    #endif
    return rays_d_tensor;
}

__device__ __constant__ float c_origin[3];
__device__ __constant__ float c_voxel_size[3];
__device__ __constant__ float c_global_domain_min[3];
__device__ __constant__ float c_global_domain_max[3];
__device__ __constant__ int c_strides[3];

__global__ void generate_query_indices_on_ray_kernel_0(const float* directions, const short *occupancy_grid, int *query_indices, short *assigned_networks,
    bool *active_ray_mask, short* depth_indices, const int num_rays, const float distance_between_points, const int max_samples_per_ray, const int max_depth_index, const float min_distance, const bool is_initial_query)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float point[3];
    int voxel_index[3];
    float current_direction[3];
    while (idx < num_rays) {
        int output_idx = idx * max_samples_per_ray;
        int output_end_idx = output_idx + max_samples_per_ray;
        bool active_ray = is_initial_query ? true : active_ray_mask[idx];
        if (active_ray) {
            for (int c = 0; c < 3; c++) {
                current_direction[c] = directions[3 * idx + c];
            }
            int depth_index = is_initial_query ? 0 : depth_indices[idx]; // potentially use depth_index from previous pass
            float distance = min_distance + depth_index * distance_between_points;
            while (depth_index < max_depth_index && output_idx < output_end_idx) {
                int voxel_flat_index = 0;
                bool inside_global_domain = true;
                for (int c = 0; c < 3; c++) {
                    point[c] = c_origin[c] + distance * current_direction[c];
                    voxel_index[c] = (point[c] - c_global_domain_min[c]) / c_voxel_size[c];
                    voxel_flat_index += voxel_index[c] * c_strides[c];
                    float epsilon = 0.001;
                    inside_global_domain = inside_global_domain && c_global_domain_min[c] + epsilon < point[c] && point[c] < c_global_domain_max[c] - epsilon;
                }
                
                short assigned_network = inside_global_domain ? occupancy_grid[voxel_flat_index] : -1; // TODO: could use 3D texture memory for this access pattern
                
                // Check if point is in empty space
                if (assigned_network != -1) {
                    assigned_networks[output_idx] = assigned_network;
                    query_indices[output_idx] = idx * max_depth_index + depth_index;
                    
                    // This code is never going to executed but it lets
                    // the compiler emits twice (!) as fast code for some reason.
                    // TODO: Investigate. Compiler bug?
                    if (point[0] == FLT_MAX) {
                        for (int c = 0; c < 3; c++) {
                            query_indices[c] = (int)point[c];
                        }
                    }
                    output_idx += 1;
                }
                depth_index += 1;
                distance += distance_between_points;
            }
            
            if (output_idx < output_end_idx) {
                // Ray already has terminated before, hence ray does not need to be reprocessed in next pass
                active_ray_mask[idx] = false;
            } else {
                // Ray casting for this ray possibly needs to be continued in next pass (given that transmittance stays above the threshold)
                active_ray_mask[idx] = true;
                depth_indices[idx] = depth_index;
            }
        }
        
        // fill remaining array with -1 (will be truncated later)
        while (output_idx < output_end_idx) {
            assigned_networks[output_idx] = -1;
            output_idx += 1;
        }
        idx += gridDim.x * blockDim.x;
    }
}

// Returns query points on ray and networks responsable for that points
std::tuple<torch::Tensor, torch::Tensor> generate_query_indices_on_ray(const torch::Tensor& origin_tensor,
    const torch::Tensor& directions_tensor,
    const torch::Tensor& occupancy_grid_tensor,
    const torch::Tensor& active_ray_mask_tensor,
    const torch::Tensor& depth_indices_tensor,
    const torch::Tensor& voxel_size_tensor,
    const torch::Tensor& global_domain_min_tensor,
    const torch::Tensor& global_domain_max_tensor,
    const torch::Tensor& strides_tensor,
    const float distance_between_points,
    const int max_samples_per_ray,
    const int max_depth_index,
    const float min_distance,
    const bool is_initial_query,
    const int64_t kernel_max_num_blocks,
    const int64_t kernel_max_num_threads,
    const int version)
{
    int num_rays = directions_tensor.size(0);
    
    torch::Tensor query_indices_tensor = torch::empty({num_rays, max_samples_per_ray}, strides_tensor.options());
    torch::Tensor assigned_networks_tensor = torch::empty({num_rays, max_samples_per_ray}, occupancy_grid_tensor.options()); // must be a short tensor
    
    float *origin = origin_tensor.data_ptr<float>();
    float *directions = directions_tensor.data_ptr<float>();
    short *occupancy_grid = occupancy_grid_tensor.data_ptr<short>();
    int *query_indices = query_indices_tensor.data_ptr<int>();
    short *assigned_networks = assigned_networks_tensor.data_ptr<short>();
    bool *active_ray_mask = active_ray_mask_tensor.data_ptr<bool>();
    short *depth_indices = depth_indices_tensor.data_ptr<short>();
    float *voxel_size = voxel_size_tensor.data_ptr<float>();
    float *global_domain_min = global_domain_min_tensor.data_ptr<float>();
    float *global_domain_max = global_domain_max_tensor.data_ptr<float>();
    int *strides = strides_tensor.data_ptr<int>();
    
    cudaMemcpyToSymbol(c_origin, origin, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_voxel_size, voxel_size, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_global_domain_min, global_domain_min, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_global_domain_max, global_domain_max, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_strides, strides, 3 * sizeof(int), 0, cudaMemcpyDeviceToDevice);
    
    int64_t num_threads, num_blocks;
    if (num_rays < kernel_max_num_threads) {
            num_threads = num_rays;
            num_blocks = 1;
    } else {
        num_threads = kernel_max_num_threads;
        
        /* computes the number of blocks that would be needed for full parallelism */
        int64_t needed_blocks = num_rays / num_threads + (num_rays % num_threads > 0);
        
        /* GPU can only excute a certain amount of blocks and therefore it is good to put an upper limit on the number of blocks */
        num_blocks = std::min(needed_blocks, kernel_max_num_blocks);
    }
    /* num_outputs unused */
    int64_t shared_memory_in_bytes = 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (version) {
        case 0:
            generate_query_indices_on_ray_kernel_0<<<num_blocks, num_threads, shared_memory_in_bytes, stream>>>(directions, occupancy_grid, query_indices, assigned_networks,
                active_ray_mask, depth_indices, num_rays, distance_between_points, max_samples_per_ray, max_depth_index, min_distance, is_initial_query);
            break;
    }
    return std::make_tuple(query_indices_tensor, assigned_networks_tensor);
}