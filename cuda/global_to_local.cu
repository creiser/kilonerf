#include "global_to_local.cuh"

#include "utils.cuh"

#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

__global__ void global_to_local_kernel(float* points, const float* domain_mins, const float* domain_maxs, const int64_t* batch_size_per_network, const int64_t* offsets) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = offsets[blockIdx.y];
    
    // Each block is only responsable for one network, hence we only have to load domain_min and domain_max per block
    // The index of the network the block is responable for is determined by blockIdx.y
    extern __shared__ float domain_min[3];
    extern __shared__ float domain_max[3];
    if (threadIdx.x < 3) {
        domain_min[threadIdx.x] = domain_mins[blockIdx.y * 3 + threadIdx.x];
    } else if (3 <= threadIdx.x && threadIdx.x < 6) {
        domain_max[threadIdx.x - 3] = domain_maxs[blockIdx.y * 3 + threadIdx.x - 3];
    }
    __syncthreads();
    
    // in-place global to local coordinate conversion 
    while (idx < 3 *  batch_size_per_network[blockIdx.y]) {
        int coord_dim = idx % 3;
        points[offset + idx] = 2 * (points[offset + idx] - domain_min[coord_dim]) / (domain_max[coord_dim] - domain_min[coord_dim]) - 1;
        idx += gridDim.x * blockDim.x;
    }
}

void global_to_local(const torch::Tensor& points_tensor, const torch::Tensor& domain_mins_tensor, const torch::Tensor& domain_maxs_tensor,
    const torch::Tensor& batch_size_per_network_tensor, int64_t kernel_num_blocks, int64_t kernel_num_threads) {
    int num_networks = batch_size_per_network_tensor.size(0);
    float *points = points_tensor.data_ptr<float>();
    float *domain_mins = domain_mins_tensor.data_ptr<float>();
    float *domain_maxs = domain_maxs_tensor.data_ptr<float>();
    int64_t *host_batch_size_per_network = batch_size_per_network_tensor.data_ptr<int64_t>();
    
    int64_t *host_offsets = new int64_t[num_networks];
    int64_t host_offset = 0;
    
    for (int i = 0; i < num_networks; ++i) {
        host_offsets[i] = host_offset;
        host_offset += 3 * host_batch_size_per_network[i];
    }
    int64_t *device_batch_size_per_network;
    int64_t *device_offsets;
    
    int64_t array_size = num_networks * sizeof(int64_t);

    cudaMalloc((void **)&device_batch_size_per_network, array_size);
    cudaMalloc((void **)&device_offsets, array_size);
    
    cudaMemcpy(device_batch_size_per_network, host_batch_size_per_network, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_offsets, host_offsets, array_size, cudaMemcpyHostToDevice);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    global_to_local_kernel<<<dim3(kernel_num_blocks, num_networks), kernel_num_threads, 0, stream>>>(points, domain_mins, domain_maxs, device_batch_size_per_network, device_offsets);
    
    delete[] host_offsets;
    cudaFree(device_batch_size_per_network);
    cudaFree(device_offsets);
}