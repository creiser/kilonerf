
#include "integrate.cuh"

#include "utils.cuh"

#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

__global__ void integrate_kernel_0(
    const float4 *rgb_sigma,
    const float* dists,
    float3 *rgb_map,
    float *acc_map,
    float *transmittance,
    bool *active_ray_mask,
    const int num_rays,
    const int samples_per_ray,
    const float transmittance_threshold,
    const bool is_initial_query)
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (ray_idx < num_rays) {
        float my_transmittance = is_initial_query ? 1.0f : transmittance[ray_idx];
        bool active_ray = my_transmittance > transmittance_threshold;
        float3 rgb_out = {0.0f, 0.0f, 0.0f};
        float acc_out = 0.0f;
        if (active_ray) {
            if (!is_initial_query) {
                rgb_out = rgb_map[ray_idx];
                acc_out = acc_map[ray_idx];
            }
            int sample_idx = ray_idx * samples_per_ray;
            int sample_end_idx = sample_idx + samples_per_ray;
            float dist = dists[ray_idx]; // distance between samples varies per ray
            while (sample_idx < sample_end_idx) {
                float4 my_rgb_sigma = rgb_sigma[sample_idx];
                float alpha = 1.0f - __expf(-my_rgb_sigma.w * dist);
                float weight = alpha * my_transmittance;
                my_transmittance *= 1.0f - alpha + 1e-10;
                rgb_out.x += my_rgb_sigma.x * weight;
                rgb_out.y += my_rgb_sigma.y * weight;
                rgb_out.z += my_rgb_sigma.z * weight;
                acc_out += weight;
                sample_idx++;
            }
            transmittance[ray_idx] = my_transmittance;
            if (my_transmittance <= transmittance_threshold) {
                active_ray_mask[ray_idx] = false;
            }
        }
        if (active_ray || is_initial_query) {
            rgb_map[ray_idx] = rgb_out;
            acc_map[ray_idx] = acc_out;
        }
        ray_idx += gridDim.x * blockDim.x;
    }
}

void integrate(const torch::Tensor& rgb_sigma_tensor, const torch::Tensor& dists_tensor, long rgb_map_pointer, const torch::Tensor& acc_map_tensor,
    const torch::Tensor& transmittance_tensor, const torch::Tensor& active_ray_mask_tensor, const int& num_rays, const int& samples_per_ray,
    const float& transmittance_threshold, const bool& is_initial_query, const int& num_blocks, const int& num_threads, const int& version)
{
    float4 *rgb_sigma = (float4*)rgb_sigma_tensor.data_ptr<float>();
    float *dists = dists_tensor.data_ptr<float>();
    float3 *rgb_map = (float3*)rgb_map_pointer;
    float *acc_map = acc_map_tensor.data_ptr<float>();
    float *transmittance = transmittance_tensor.data_ptr<float>();
    bool *active_ray_mask = active_ray_mask_tensor.data_ptr<bool>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (version) {
        case 0:
            integrate_kernel_0<<<num_blocks, num_threads, 0, stream>>>(rgb_sigma, dists, rgb_map, acc_map, transmittance,
                active_ray_mask, num_rays, samples_per_ray, transmittance_threshold, is_initial_query);
            break;
    }
    gpuErrchk( cudaPeekAtLastError() );
    #ifdef DEBUG_SYNC
    gpuErrchk( cudaDeviceSynchronize() );
    #endif
}

__device__ __constant__ float3 c_background_color;
__global__ void replace_transparency_by_background_color_kernel(float3 *rgb_map, float *acc_map, int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < num_pixels) {
        float3 in = rgb_map[idx];
        float t = 1.0f - acc_map[idx];
        float3 out;
        out.x = in.x + c_background_color.x * t;
        out.y = in.y + c_background_color.y * t;
        out.z = in.z + c_background_color.z * t;
        rgb_map[idx] = out;
        idx += gridDim.x * blockDim.x;
    }
}

void replace_transparency_by_background_color(long rgb_map_pointer, const torch::Tensor& acc_map_tensor, const torch::Tensor& background_color_tensor, const int& num_blocks, const int& num_threads)
{
    float3 *rgb_map = (float3*)rgb_map_pointer;
    float *acc_map = acc_map_tensor.data_ptr<float>();
    float3 *background_color = (float3*)background_color_tensor.data_ptr<float>();
    cudaMemcpyToSymbol(c_background_color, background_color, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    int num_pixels = acc_map_tensor.size(0) * acc_map_tensor.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    replace_transparency_by_background_color_kernel<<<num_blocks, num_threads, 0, stream>>>(rgb_map, acc_map, num_pixels);
    gpuErrchk( cudaPeekAtLastError() );
    #ifdef DEBUG_SYNC
    gpuErrchk( cudaDeviceSynchronize() );
    #endif
}

