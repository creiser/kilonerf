#ifndef __MULTIMATMUL__
#define __MULTIMATMUL__

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <torch/extension.h>

void init_stream_pool(int64_t num_streams);
void destroy_stream_pool();
void init_magma();

int init_multimatmul_magma_grouped(int64_t num_networks, int64_t out_features, int64_t in_features, std::vector<int> group_limits);
torch::Tensor multimatmul_magma_grouped_static_without_bias(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index);
torch::Tensor multimatmul_magma_grouped_static(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index);
torch::Tensor multimatmul_magma_grouped_static_without_bias_transposed_weights(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index);
void deinit_multimatmul_magma_grouped(int aux_index);

// Required for backward pass of MultiNetworkLinear layer.
torch::Tensor multi_row_sum_reduction(const torch::Tensor& input_matrix_tensor, const torch::Tensor& batch_size_per_network_tensor);
torch::Tensor multimatmul_A_transposed(const torch::Tensor& A_tensor, const torch::Tensor& B_tensor, const torch::Tensor& batch_size_per_network_tensor);

#endif
