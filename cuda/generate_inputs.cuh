#ifndef __GENERATE_INPUTS__
#define __GENERATE_INPUTS__

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <torch/extension.h>

torch::Tensor get_rays_d(const int& H, const int& W, const float& cx, const float& cy,  const float& fx, const float& fy, const torch::Tensor& c2w_tensor, const int& root_num_blocks, const int& root_num_threads);

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
    const int version);

#endif 