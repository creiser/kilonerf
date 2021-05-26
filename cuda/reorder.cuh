#ifndef __REORDER__
#define __REORDER__

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <torch/extension.h>

torch::Tensor gather_int32(const torch::Tensor& map_tensor, const torch::Tensor& input_tensor);
void sort_by_key_int16_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor);
void sort_by_key_int16_int32(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor);
torch::Tensor scatter_int32_float4(const torch::Tensor& map_tensor, const torch::Tensor& input_tensor);

#endif