#include "fourier_features.cuh"

#include "utils.cuh"

#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

template <typename T>
__global__ void fourier_features_kernel(const T* input, const T* frequency_bands, T* output, const unsigned num_inputs,
    const unsigned num_outputs, const unsigned num_outputs_per_input, const unsigned num_frequencies) {
    int input_index = threadIdx.x + blockDim.x * blockIdx.x;
    int input_start_index = blockDim.x * blockIdx.x; /* indices the first input that this thread block processes */
    
    extern __shared__ T frequency_bands_shared[];
    T *output_shared = &frequency_bands_shared[num_frequencies]; 
    
    // Load frequency_bands into shared memory
    if (threadIdx.x < num_frequencies) {
        frequency_bands_shared[threadIdx.x] = frequency_bands[threadIdx.x];
    }
    __syncthreads();
    
    while (input_index < num_inputs) {
        int output_index = threadIdx.x * num_outputs_per_input;
        
        // Coalesced read from global memory
        T input_val = input[input_index];
        
        // Uncoalesced writes into shared memory
        output_shared[output_index] = input_val;
        output_index += 1;
        for (int r = 0; r < num_frequencies; r++) {
            output_shared[output_index] = cosf(frequency_bands_shared[r] * input_val);
            output_index += 1;
        }
        for (int r = 0; r < num_frequencies; r++) {
            output_shared[output_index] = sinf(frequency_bands_shared[r] * input_val);
            output_index += 1;
        }
        
        __syncthreads();
        
        // Coaleseced write from shared memory into global memory
        int global_offset = input_start_index * num_outputs_per_input;
        T* output_offseted = &output[global_offset];
        output_index = threadIdx.x;
        int active_num_threads_in_block = min(num_inputs - input_start_index, blockDim.x);  // needed for last blocks where not all threads are active
        int effective_num_outputs_in_block = active_num_threads_in_block * num_outputs_per_input;
        while (output_index < effective_num_outputs_in_block) {
            output_offseted[output_index] = output_shared[output_index];
            output_index += active_num_threads_in_block;
        }
        
        __syncthreads();
        
        input_index += gridDim.x * blockDim.x;
        input_start_index += gridDim.x * blockDim.x;
    }
}

void fourier_features_kernel_wrapper(const float* input, const float* frequency_bands, float* output, const unsigned num_inputs,
    const unsigned num_outputs, const unsigned num_outputs_per_input, const unsigned num_frequencies,
    int64_t kernel_max_num_blocks, int64_t kernel_max_num_threads, std::string implementation, cudaStream_t& stream) {

    int64_t num_threads;
    int64_t num_blocks;
    
    if (num_inputs < kernel_max_num_threads) {
        num_threads = num_inputs;
        num_blocks = 1;
    } else {
        num_threads = kernel_max_num_threads;
        
        /* computes the number of blocks that would be needed for full parallelism */
        int64_t needed_blocks = num_inputs / num_threads + (num_inputs % num_threads > 0);
        
        /* GPU can only excute a certain amount of blocks and therefore it is good to put an upper limit on the number of blocks */
        num_blocks = std::min(needed_blocks, kernel_max_num_blocks);
    }
    /* num_outputs unused */
    int64_t shared_memory_in_bytes = (num_frequencies + num_threads * num_outputs_per_input) * sizeof(float);
    fourier_features_kernel<<<num_blocks, num_threads, shared_memory_in_bytes, stream>>>(input, frequency_bands, output, num_inputs, num_outputs, num_outputs_per_input, num_frequencies);
    
}

torch::Tensor compute_fourier_features(const torch::Tensor& input_tensor, const torch::Tensor& frequency_bands_tensor, int64_t kernel_max_num_blocks, int64_t kernel_max_num_threads, std::string implementation) {
    const unsigned num_frequencies = frequency_bands_tensor.size(0);
    const unsigned num_inputs = input_tensor.size(0);
    const unsigned num_outputs_per_input = 2 * num_frequencies + 1;
    const unsigned num_outputs = num_inputs * num_outputs_per_input; 
    
    torch::Tensor output_tensor = torch::empty({num_outputs}, input_tensor.options());
    
    float *input = input_tensor.data_ptr<float>();
    float *frequency_bands = frequency_bands_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fourier_features_kernel_wrapper(input, frequency_bands, output, num_inputs, num_outputs, num_outputs_per_input, num_frequencies, kernel_max_num_blocks, kernel_max_num_threads, implementation, stream);
    return output_tensor;
}