#ifndef __NETWORK_EVAL__
#define __NETWORK_EVAL__

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <torch/extension.h>

torch::Tensor network_eval_query_index(const torch::Tensor& query_indices_tensor, const torch::Tensor& params_tensor,
    const torch::Tensor& domain_mins_tensor, const torch::Tensor& domain_maxs_tensor, const torch::Tensor& starts_tensor, const torch::Tensor& ends_tensor, const torch::Tensor& origin_tensor,
    const torch::Tensor& c2w_tensor,
    const int& num_networks, const int& hidden_dim,
    const int& H, const int& W, const float& cx, const float& cy, const float& fx, const float& fy, const int& max_depth_index, const float& min_distance, const float& distance_between_samples,
    const int& num_blocks, const int& num_threads, const int& version);

#endif
