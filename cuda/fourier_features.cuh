#ifndef __FOURIER_FEATURES__
#define __FOURIER_FEATURES__

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <torch/extension.h>

torch::Tensor compute_fourier_features(const torch::Tensor& input_tensor, const torch::Tensor& frequency_bands_tensor, int64_t kernel_max_num_blocks, int64_t kernel_max_num_threads, std::string implementation);

#endif