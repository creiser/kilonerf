#include "network_eval.cuh"

#include "utils.cuh"

#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

constexpr int calculate_fourier_embedding_num_output_channels(int num_input_channels, int num_frequencies)
{
    return num_input_channels * (2 * num_frequencies + 1);
}

__device__ __forceinline__ float sigmoid (float x)
{
    return 1.0 / (1.0 + __expf (-x));
}

__device__ __constant__ float c_c2w[9];
__device__ __constant__ float c_origin[3];

// Hardcoded number of frequencies for Fourier features:
// Position:   10 frequencies
// Direction:   4 frequencies
template<int hidden_dim>
__global__ void network_eval_query_index_kernel_0(const int* query_indices, const float* params,
    const float *domain_mins, const float *domain_maxs, const int *starts, const int *ends, float *output_vectors,
    const int H, const int W, const float cx, const float cy, const float fx, const float fy, const int max_depth_index, const float min_distance, const float distance_between_samples)
{
    constexpr int num_frequencies_position = 10;
    constexpr int num_frequencies_direction = 4;
    
    constexpr int pos_dim = 3;
    constexpr int dir_dim = 3;
    constexpr int pos_embed_dim = calculate_fourier_embedding_num_output_channels(pos_dim, num_frequencies_position);
    constexpr int dir_embed_dim = calculate_fourier_embedding_num_output_channels(dir_dim, num_frequencies_direction);
    constexpr int rgb_dim = 3;
    constexpr int out_dim = 4;
    
    constexpr float frequency_bands[10] = {1., 2., 4., 8., 16., 32., 64., 128., 256., 512.};
    
    // for each layer: (#inputs + 1) * #outputs
    constexpr int param_size = (pos_embed_dim + 1) * hidden_dim +
                               (hidden_dim + 1) * hidden_dim +
                               (hidden_dim + 1) * (hidden_dim + 1) +
                               (hidden_dim + dir_embed_dim + 1) * hidden_dim + 
                               (hidden_dim + 1) * rgb_dim;
                              
    if (starts[blockIdx.x] == ends[blockIdx.x]) {
        return; // assiociated network does not need to be queried
    }
    
    // Each block is only responsable for one network, hence we only have to load a single domain_min and domain_max per block
    // TODO: could also calculate those from global_domain_min and global_domain_max (in constant cache)
    extern __shared__ float domain_min[3];
    extern __shared__ float domain_max[3];
    if (threadIdx.x < 3) {
        domain_min[threadIdx.x] = domain_mins[blockIdx.x * 3 + threadIdx.x];
    } else if (3 <= threadIdx.x && threadIdx.x < 6) {
        domain_max[threadIdx.x - 3] = domain_maxs[blockIdx.x * 3 + threadIdx.x - 3];
    }
                         
    // Load single network into shared memory
    __shared__ float network_cache[param_size];
    int load_idx = threadIdx.x;
    int network_offset = blockIdx.x * param_size; // block i is reponsible for network i
    while (load_idx < param_size) {
        network_cache[load_idx] = params[network_offset + load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();
                    
    int idx = starts[blockIdx.x] + threadIdx.x;
    while (idx < ends[blockIdx.x]) {
        // Compute position and direction from query index
        // Memory layout: (y * W + x) * max_depth_index + depth_index;
        // H x W x D
        int y = query_indices[idx];
        int depth_index = y % max_depth_index;
        y /= max_depth_index;
        int x = y % W;
        y /= W;
        
        // Calculate unnormalized direction based on camera transform matrix c_c2w and pixel position (x, y)
        float in[3] = {((float)x - cx) / fx, -((float)y - cy) / fy, -1};
        float direction[3] = {0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                direction[i] += in[j] * c_c2w[i * 3 + j];
            }
        }
        
        // Compute position with help of ray origin, ray direction (unnormalized!) and depth index
        float distance = min_distance + depth_index * distance_between_samples;
        float position[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            position[i] = c_origin[i] + distance * direction[i];
        }
        
        // Normalize direction
        float norm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            norm += direction[i] * direction[i];
        }
        norm = sqrtf(norm);
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            direction[i] /= norm;
        }
        
        // Actual network query
        int param_offset = 0;
        
        // First layer
        float my_hidden_vector_0[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_0[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float input_elem = 2 * (position[j] - domain_min[j]) / (domain_max[j] - domain_min[j]) - 1; // global to local conversion
        
            // Fourier feature
            #pragma unroll
            for (int e = 0; e < 2 * num_frequencies_position + 1; e++) {
                float embedded_input_elem;
                if (e == 0) {
                    embedded_input_elem = input_elem;
                }
                else if (e < num_frequencies_position + 1) {
                    embedded_input_elem = __cosf(frequency_bands[e - 1] *  input_elem);
                }
                else {
                    embedded_input_elem = __sinf(frequency_bands[e - (num_frequencies_position + 1)] * input_elem);
                }
                
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        
        // Second layer
        float my_hidden_vector_1[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_1[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_1[i] += fmaxf(my_hidden_vector_0[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        
        // Third layer: outputs density + feature
        float my_hidden_vector_2[hidden_dim + 1]; // 0-th entry of this vector is the density
        #pragma unroll
        for (int i = 0; i < hidden_dim + 1; i++) {
            my_hidden_vector_2[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim + 1; i++) {
                my_hidden_vector_2[i] += fmaxf(my_hidden_vector_1[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }

        // Fourth layer: #inputs = hidden_dim + dir_embed_dim
        float my_hidden_vector_3[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_3[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        // Features
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                // offset + 1 because of density, no ReLU (previous layer)
                my_hidden_vector_3[i] += my_hidden_vector_2[j + 1] * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        // Embedded directions
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float input_elem = direction[j];
        
            // Fourier feature
            #pragma unroll
            for (int e = 0; e < 2 * num_frequencies_direction + 1; e++) {
                float embedded_input_elem;
                if (e == 0) {
                    embedded_input_elem = input_elem;
                }
                else if (e < num_frequencies_direction + 1) {
                    embedded_input_elem = __cosf(frequency_bands[e - 1] *  input_elem);
                }
                else {
                    embedded_input_elem = __sinf(frequency_bands[e - (num_frequencies_direction + 1)] * input_elem);
                }
                
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    // offset + 1 because of density, no ReLU (previous layer)
                    my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        
        // Fifth layer
        float my_rgb_vector[rgb_dim];
        #pragma unroll
        for (int i = 0; i < rgb_dim; i++) {
            my_rgb_vector[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < rgb_dim; i++) {
                my_rgb_vector[i] += fmaxf(my_hidden_vector_3[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        #pragma unroll
        for (int i = 0; i < out_dim; i++) {
            float output_elem;
            if (i < rgb_dim) {
                output_elem = sigmoid(my_rgb_vector[i]); // RGB + Sigmoid
            } else {
                output_elem = fmaxf(my_hidden_vector_2[0], 0.0f); // Density + ReLU
            }
            output_vectors[idx * out_dim + i] = output_elem; 
        }
        idx += blockDim.x;
    }
}

torch::Tensor network_eval_query_index(const torch::Tensor& query_indices_tensor, const torch::Tensor& params_tensor,
    const torch::Tensor& domain_mins_tensor, const torch::Tensor& domain_maxs_tensor, const torch::Tensor& starts_tensor, const torch::Tensor& ends_tensor, const torch::Tensor& origin_tensor,
    const torch::Tensor& c2w_tensor,
    const int& num_networks, const int& hidden_dim,
    const int& H, const int& W, const float& cx, const float& cy, const float& fx, const float& fy, const int& max_depth_index, const float& min_distance, const float& distance_between_samples,
    const int& num_blocks, const int& num_threads, const int& version)
{
    int *query_indices = query_indices_tensor.data_ptr<int>();
    float *params = params_tensor.data_ptr<float>();
    float *domain_mins = domain_mins_tensor.data_ptr<float>();
    float *domain_maxs = domain_maxs_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();
    float *origin = origin_tensor.data_ptr<float>();
    float *c2w = c2w_tensor.data_ptr<float>();
    int batch_size = query_indices_tensor.size(0);
    torch::Tensor output_vectors_tensor = torch::ones({batch_size, 4}, params_tensor.options()); //  TODO: check if "ones" can be replaced by "empty" here
    float *output_vectors = output_vectors_tensor.data_ptr<float>();
    
    cudaMemcpyToSymbol(c_origin, origin, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_c2w, c2w, 9 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (version) {
        case 0:
            switch (hidden_dim) {
                case 32:
                    network_eval_query_index_kernel_0<32><<<num_networks, num_threads, 0, stream>>>(query_indices, params, domain_mins, domain_maxs, starts, ends, output_vectors,
                        H, W, cx, cy, fx, fy, max_depth_index, min_distance, distance_between_samples);
                    break;
                default:
                    printf("Unsupported hidden_dim: %d\n", hidden_dim);  
            }
            break;

    }
    gpuErrchk( cudaPeekAtLastError() );
    #ifdef DEBUG_SYNC
    gpuErrchk( cudaDeviceSynchronize() );
    #endif  
    return output_vectors_tensor;
}