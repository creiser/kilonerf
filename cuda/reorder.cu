
#include "reorder.cuh"

#include "utils.cuh"

#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

torch::Tensor gather_int32(const torch::Tensor& map_tensor, const torch::Tensor& input_tensor)
{
    int output_size = map_tensor.size(0);
    torch::Tensor output_tensor = torch::empty({output_size}, input_tensor.options());
    int32_t *map = map_tensor.data_ptr<int32_t>();
    int32_t *input = input_tensor.data_ptr<int32_t>();
    int32_t *output = output_tensor.data_ptr<int32_t>();
    thrust::gather(thrust::device, map, map + output_size, input, output);
    return output_tensor;
}

void sort_by_key_int16_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor)
{
    int length = keys_tensor.size(0);
    int16_t *keys = keys_tensor.data_ptr<int16_t>();
    int64_t *values = values_tensor.data_ptr<int64_t>();
    thrust::sort_by_key(thrust::device, keys, keys + length, values);
}

void sort_by_key_int16_int32(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor)
{
    int length = keys_tensor.size(0);
    int16_t *keys = keys_tensor.data_ptr<int16_t>();
    int32_t *values = values_tensor.data_ptr<int32_t>();
    thrust::sort_by_key(thrust::device, keys, keys + length, values);
}

torch::Tensor scatter_int32_float4(const torch::Tensor& map_tensor, const torch::Tensor& input_tensor)
{
    int input_size = input_tensor.size(0);
    torch::Tensor output_tensor = torch::empty({input_size, 4}, input_tensor.options());
    int32_t *map = map_tensor.data_ptr<int32_t>();
    float4 *input = (float4*)input_tensor.data_ptr<float>();
    float4 *output = (float4*)output_tensor.data_ptr<float>();
    thrust::scatter(thrust::device, input, input + input_size, map, output);
    return output_tensor;
}
