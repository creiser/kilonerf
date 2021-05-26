#ifndef __INTEGRATE__
#define __INTEGRATE__

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <torch/extension.h>

void integrate(const torch::Tensor& rgb_sigma_tensor, const torch::Tensor& dists_tensor, long rgb_map, const torch::Tensor& acc_map_tensor,
    const torch::Tensor& transmittance_tensor, const torch::Tensor& active_ray_mask_tensor, const int& num_rays, const int& samples_per_ray,
    const float& transmittance_threshold, const bool& is_initial_query, const int& num_blocks, const int& num_threads, const int& version);
    
void replace_transparency_by_background_color(long rgb_map_pointer, const torch::Tensor& acc_map_tensor, const torch::Tensor& background_color_tensor, const int& num_blocks, const int& num_threads);

#endif