#ifndef __GLOBAL_TO_LOCAL__
#define __GLOBAL_TO_LOCAL__

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <torch/extension.h>

void global_to_local(const torch::Tensor& points_tensor, const torch::Tensor& domain_mins_tensor, const torch::Tensor& domain_maxs_tensor,
    const torch::Tensor& batch_size_per_network_tensor, int64_t kernel_num_blocks, int64_t kernel_num_threads);

#endif