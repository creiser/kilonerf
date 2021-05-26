
#include "multimatmul.cuh"

#include "utils.cuh"

#include <ATen/cuda/CUDABlas.h>
using namespace torch::indexing;

#include <iostream>
#include <algorithm>
#include <map>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include "magma_v2.h"

class StreamPool
{
    public:
        static void init(int64_t num_streams);
        static void destroy();
        static cudaStream_t get_next_stream();

        static cudaStream_t *streamArray;
        static int64_t num_streams;
    protected:
       static int64_t counter;
};

cudaStream_t *StreamPool::streamArray = nullptr;
int64_t StreamPool::counter = 0;
int64_t StreamPool::num_streams = 0;

void StreamPool::init(int64_t num_streams)
{
    StreamPool::streamArray = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t *));
    StreamPool::counter = 0;
    StreamPool::num_streams = num_streams;
    for (int i = 0; i < num_streams ; i++)
    {
        cudaError_t cudaErr = cudaStreamCreate(&StreamPool::streamArray[i]);
        if (cudaErr != cudaSuccess)
        {
            std::cerr << "Cannot create stream." << std::endl;
        }
    }
}

void StreamPool::destroy()
{
    free(streamArray);
}

cudaStream_t StreamPool::get_next_stream()
{
    cudaStream_t current_stream = StreamPool::streamArray[counter];
    counter = (counter + 1) % num_streams;
    return current_stream;
}

void init_stream_pool(int64_t num_streams)
{
    StreamPool::init(num_streams);
}

void destroy_stream_pool()
{
    StreamPool::destroy();
}

magma_queue_t magma_queue = NULL;
magma_queue_t *magma_streams = NULL;

void init_magma()
{
    magma_init(); 
    magma_int_t dev = 0;
    
    cudaStream_t cuda_stream = at::cuda::getDefaultCUDAStream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    magma_queue_create_from_cuda(
        dev,
     	cuda_stream,
     	cublas_handle,
     	NULL,
     	&magma_queue
    );
    
    magma_streams = new magma_queue_t[StreamPool::num_streams];
    for (int i = 0; i < StreamPool::num_streams; ++i) {
            magma_queue_create_from_cuda(
                dev,
             	StreamPool::streamArray[i],
             	cublas_handle,
             	NULL,
             	&magma_streams[i]
            );
    }
}

// Important: Assumes row major, gridDim.x * blockDim.x must be a multiple of vector_length
__global__ void multi_expand_kernel(const float* vector, const int vector_length, float* matrix, const int* matrix_num_rows, const int* output_offsets)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int input_offset = blockIdx.y * vector_length;
    int output_offset = output_offsets[blockIdx.y];
    float myval = vector[input_offset + idx % vector_length];
    int matrix_size = matrix_num_rows[blockIdx.y] * vector_length;
    while (idx < matrix_size) {
        matrix[output_offset + idx] = myval;
        idx += gridDim.x * blockDim.x;
    }
}

void multi_expand_kernel_wrapper(float *vector, int vector_length, float* matrix, const int* matrix_num_rows, const int* output_offsets, 
    int num_vectors, int num_blocks, int kernel_max_num_threads, cudaStream_t& stream)
{
    /* ensures that num_threads is the biggest multiple of vector_length smaller than kernel_max_num_threads
       this is nessecary because of our specific kernel implementation */
    int num_threads = (kernel_max_num_threads / vector_length) * vector_length;
    multi_expand_kernel<<<dim3(num_blocks, num_vectors), num_threads, 0 , stream>>>(vector, vector_length, matrix, matrix_num_rows, output_offsets);
}

struct multimatmul_magma_grouped_aux_data
{
    magma_int_t* batch_count_per_group;
    
    float **h_A_array = NULL;
    float **h_B_array = NULL;
    float **h_C_array = NULL;
    float **d_A_array = NULL;
    float **d_B_array = NULL;
    float **d_C_array = NULL;
        
    magma_int_t *h_M, *h_N, *h_K; 
    magma_int_t *d_M, *d_N, *d_K;
     
    magma_int_t *h_ldda, *d_ldda;
    magma_int_t *h_lddb, *d_lddb;
    magma_int_t *h_lddc, *d_lddc;
    
    magma_int_t *h_output_offsets, *d_output_offsets;
    magma_int_t *h_matrix_num_rows, *d_matrix_num_rows;
    
    magma_int_t *max_m, *max_n, *max_k;
};

std::map<int, multimatmul_magma_grouped_aux_data> multimatmul_magma_grouped_aux_data_array;
int multimatmul_magma_grouped_aux_data_counter = 0;

int init_multimatmul_magma_grouped(int64_t num_networks, int64_t out_features, int64_t in_features, std::vector<int> group_limits)
{
    multimatmul_magma_grouped_aux_data aux_data;

    int64_t matrixM = out_features;
    //int64_t matrixN; // gets set to batch_size_per_network
    int64_t matrixK = in_features;
    
    int64_t rowsA = matrixM;
    int64_t rowsB = matrixK;
    int64_t rowsC = matrixM;

    int num_groups = group_limits.size() + 1;
    magma_int_t maxBatchCount = num_networks;

    // Allocate host and device memory for auxilary structures
    magma_imalloc_cpu(&aux_data.batch_count_per_group, num_groups);
    
    // sizes on the cpu
    magma_imalloc_cpu(&aux_data.h_M, num_groups*(maxBatchCount+1));
    magma_imalloc_cpu(&aux_data.h_N, num_groups*(maxBatchCount+1));
    magma_imalloc_cpu(&aux_data.h_K, num_groups*(maxBatchCount+1));
    // size arrays on the GPU should be at least of size (batchCount+1)
    magma_imalloc(&aux_data.d_M, num_groups*(maxBatchCount+1));
    magma_imalloc(&aux_data.d_N, num_groups*(maxBatchCount+1));
    magma_imalloc(&aux_data.d_K, num_groups*(maxBatchCount+1));
    
    // allocate space for the leading dim
    magma_imalloc_cpu(&aux_data.h_ldda, num_groups*(maxBatchCount+1));
    magma_imalloc_cpu(&aux_data.h_lddb, num_groups*(maxBatchCount+1));
    magma_imalloc_cpu(&aux_data.h_lddc, num_groups*(maxBatchCount+1));
    // leading dimension arrays on the GPU should be at least of size (batchCount+1)
    magma_imalloc(&aux_data.d_ldda, num_groups*(maxBatchCount+1));
    magma_imalloc(&aux_data.d_lddb, num_groups*(maxBatchCount+1));
    magma_imalloc(&aux_data.d_lddc, num_groups*(maxBatchCount+1));
    
    // for multi_expand
    magma_imalloc_cpu(&aux_data.h_output_offsets, maxBatchCount+1);
    magma_imalloc(&aux_data.d_output_offsets, maxBatchCount+1);
    
    magma_imalloc_cpu(&aux_data.h_matrix_num_rows, maxBatchCount+1);
    magma_imalloc(&aux_data.d_matrix_num_rows, maxBatchCount+1);

    
    // pointer arrays
    magma_malloc_cpu((void**)&aux_data.h_A_array, num_groups*(maxBatchCount+1)*sizeof(float*));
    magma_malloc_cpu((void**)&aux_data.h_B_array, num_groups*(maxBatchCount+1)*sizeof(float*));
    magma_malloc_cpu((void**)&aux_data.h_C_array, num_groups*(maxBatchCount+1)*sizeof(float*));
    
    magma_malloc((void**)&aux_data.d_A_array, num_groups*(maxBatchCount+1)*sizeof(float*));
    magma_malloc((void**)&aux_data.d_B_array, num_groups*(maxBatchCount+1)*sizeof(float*));
    magma_malloc((void**)&aux_data.d_C_array, num_groups*(maxBatchCount+1)*sizeof(float*));
    
    magma_imalloc_cpu(&aux_data.max_m, num_groups);
    magma_imalloc_cpu(&aux_data.max_n, num_groups);
    magma_imalloc_cpu(&aux_data.max_k, num_groups);

    // i refers to the group
    #define GROUP_IDX(i,j) i*(maxBatchCount+1)+j
    
    for (int i = 0; i < num_groups; i++) {
       aux_data.max_m[i] = matrixM;
       aux_data.max_k[i] = matrixK;
    }

    for (int i = 0; i < num_networks; ++i) {
        for (int group = 0; group < num_groups; ++group)
        {
            int dest_idx = GROUP_IDX(group, i);
            aux_data.h_M[dest_idx] = matrixM;
            aux_data.h_K[dest_idx] = matrixK;
            
            aux_data.h_ldda[dest_idx] = rowsA;
            aux_data.h_lddb[dest_idx] = rowsB;
            aux_data.h_lddc[dest_idx] = rowsC;
        }
    }
    
    // Copy auxilary data from host to device
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(magma_int_t), aux_data.h_M, 1, aux_data.d_M, 1, magma_queue );
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(magma_int_t), aux_data.h_K, 1, aux_data.d_K, 1, magma_queue );
    
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(magma_int_t), aux_data.h_ldda, 1, aux_data.d_ldda, 1, magma_queue );
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(magma_int_t), aux_data.h_lddb, 1, aux_data.d_lddb, 1, magma_queue );
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(magma_int_t), aux_data.h_lddc, 1, aux_data.d_lddc, 1, magma_queue );
    
    int aux_index = multimatmul_magma_grouped_aux_data_counter;
    multimatmul_magma_grouped_aux_data_array[aux_index] = aux_data;
    multimatmul_magma_grouped_aux_data_counter += 1;
    return aux_index;
}

torch::Tensor _multimatmul_magma_grouped_static(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index, bool transpose_weights, bool use_bias)
{
    multimatmul_magma_grouped_aux_data aux_data = multimatmul_magma_grouped_aux_data_array[aux_index];

    int64_t batch_size = input_vectors.size(0);
    int64_t num_networks = batch_size_per_network.sizes()[0];
    
    torch::Tensor result = torch::empty({batch_size, out_features}, input_vectors.options());
    
    float alpha = MAGMA_S_MAKE(  1.0f, 0.f );
    float beta  = MAGMA_S_MAKE( use_bias ? 1.0f : 0.0f,  0.0f );
    
    int64_t matrixN; // gets set to batch_size_per_network

    int64_t weight_matrix_size = in_features * out_features;
    
    int64_t input_vectors_index = 0;
    int64_t results_index = 0;
    
    float* weights_ptr = weights.data_ptr<float>();
    float* input_vectors_ptr = input_vectors.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    float* biases_ptr = biases.data_ptr<float>();
    
    int64_t *batch_size_per_network_ptr = batch_size_per_network.data_ptr<int64_t>();
    
    int num_groups = group_limits.size() + 1;
    int* group_limits_array = &group_limits[0];
    magma_int_t maxBatchCount = num_networks;
    
    
    // i refers to the group
    #define GROUP_IDX(i,j) i*(maxBatchCount+1)+j
    
    for (int i = 0; i < num_groups; i++) {
       aux_data.batch_count_per_group[i] = 0;
       aux_data.max_n[i] = 0;
   }

    for (int i = 0; i < num_networks; ++i) {
        matrixN = batch_size_per_network_ptr[i];
        
        if (use_bias) {
            aux_data.h_output_offsets[i] = results_index;
            aux_data.h_matrix_num_rows[i] = matrixN;
        }
        
        if (matrixN > 0) {
            // determine right group according to size
            int group = num_groups - 1;
            for (int j = 0; j < num_groups - 1; ++j) {
                if (matrixN > group_limits_array[j]) {
                    group = j;
                    break;
                }
            }
            int dest_idx = GROUP_IDX(group, aux_data.batch_count_per_group[group]);
        
            aux_data.h_N[dest_idx] = matrixN;

            aux_data.h_A_array[dest_idx] = &weights_ptr[i * weight_matrix_size];
            aux_data.h_B_array[dest_idx] = &input_vectors_ptr[input_vectors_index];
            aux_data.h_C_array[dest_idx] = &result_ptr[results_index];
            
            aux_data.batch_count_per_group[group] += 1;
            if (matrixN > aux_data.max_n[group])
                aux_data.max_n[group] = matrixN;
        }
        
        input_vectors_index += matrixN * in_features;
        results_index += matrixN * out_features;
    }
    
    
    // Copy auxilary data from host to device
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(magma_int_t), aux_data.h_N, 1, aux_data.d_N, 1, magma_queue );

    // for multi_expand
    if (use_bias) {
        magma_setvector(maxBatchCount, sizeof(magma_int_t), aux_data.h_output_offsets, 1, aux_data.d_output_offsets, 1, magma_queue );
        magma_setvector(maxBatchCount, sizeof(magma_int_t), aux_data.h_matrix_num_rows, 1, aux_data.d_matrix_num_rows, 1, magma_queue );
    }
    
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(float*), aux_data.h_A_array, 1, aux_data.d_A_array, 1, magma_queue );
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(float*), aux_data.h_B_array, 1, aux_data.d_B_array, 1, magma_queue );
    magma_setvector(num_groups*(maxBatchCount+1), sizeof(float*), aux_data.h_C_array, 1, aux_data.d_C_array, 1, magma_queue );

    
    if (use_bias) {
        cudaStream_t stream = at::cuda::getDefaultCUDAStream();
        multi_expand_kernel_wrapper(biases_ptr, out_features, result_ptr, aux_data.d_matrix_num_rows, aux_data.d_output_offsets,
            maxBatchCount, kernel_num_blocks, kernel_num_threads, stream);
    }
    
    // We can transpose weights at zero cost here
    magma_trans_t trans_A_op = transpose_weights ? MagmaTrans : MagmaNoTrans;
    magma_int_t *d_ldda = transpose_weights ? aux_data.d_K : aux_data.d_M;
    
    int current_stream_count = 0;
    for (int i = 0; i < num_groups; ++i) {
        if (aux_data.batch_count_per_group[i] > 0) {
            int group_offset = GROUP_IDX(i, 0);
            magmablas_sgemm_vbatched_max_nocheck(trans_A_op, MagmaNoTrans,
                                 &aux_data.d_M[group_offset], &aux_data.d_N[group_offset], &aux_data.d_K[group_offset],
                                 alpha, &aux_data.d_A_array[group_offset], &d_ldda[group_offset],
                                        &aux_data.d_B_array[group_offset], &aux_data.d_lddb[group_offset],
                                 beta,  &aux_data.d_C_array[group_offset], &aux_data.d_lddc[group_offset],
                                 aux_data.batch_count_per_group[i],
                                 aux_data.max_m[i], aux_data.max_n[i], aux_data.max_k[i],
                                 magma_streams[current_stream_count]);
           current_stream_count = (current_stream_count + 1) % StreamPool::num_streams;
         }
     }
    
    return result;
}

torch::Tensor multimatmul_magma_grouped_static_without_bias(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index)
{
    return _multimatmul_magma_grouped_static(biases, input_vectors, weights, out_features, in_features, batch_size_per_network,
        kernel_num_blocks, kernel_num_threads, group_limits, aux_index, false, false);
}

torch::Tensor multimatmul_magma_grouped_static(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index)
{
    return _multimatmul_magma_grouped_static(biases, input_vectors, weights, out_features, in_features, batch_size_per_network,
        kernel_num_blocks, kernel_num_threads, group_limits, aux_index, false, true);
}

torch::Tensor multimatmul_magma_grouped_static_without_bias_transposed_weights(const torch::Tensor& biases, const torch::Tensor& input_vectors, const torch::Tensor& weights,
    int64_t out_features, int64_t in_features, const torch::Tensor& batch_size_per_network, int64_t kernel_num_blocks, int64_t kernel_num_threads,
    std::vector<int> group_limits, int aux_index)
{
    return _multimatmul_magma_grouped_static(biases, input_vectors, weights, out_features, in_features, batch_size_per_network,
        kernel_num_blocks, kernel_num_threads, group_limits, aux_index, true, false);
}

void deinit_multimatmul_magma_grouped(int aux_index)
{            
    multimatmul_magma_grouped_aux_data aux_data = multimatmul_magma_grouped_aux_data_array[aux_index];
    
    // Free memory
    magma_free_cpu(aux_data.batch_count_per_group);
    
    magma_free_cpu(aux_data.h_M);
    magma_free_cpu(aux_data.h_N);
    magma_free_cpu(aux_data.h_K);

    magma_free(aux_data.d_M);
    magma_free(aux_data.d_N);
    magma_free(aux_data.d_K);

    magma_free_cpu(aux_data.h_ldda);
    magma_free_cpu(aux_data.h_lddb);
    magma_free_cpu(aux_data.h_lddc);

    magma_free(aux_data.d_ldda);
    magma_free(aux_data.d_lddb);
    magma_free(aux_data.d_lddc);
    
    magma_free_cpu(aux_data.h_output_offsets);
    magma_free(aux_data.d_output_offsets);
    magma_free_cpu(aux_data.h_matrix_num_rows);
    magma_free(aux_data.d_matrix_num_rows);

    magma_free_cpu(aux_data.h_A_array);
    magma_free_cpu(aux_data.h_B_array);
    magma_free_cpu(aux_data.h_C_array);
    
    magma_free(aux_data.d_A_array);
    magma_free(aux_data.d_B_array);
    magma_free(aux_data.d_C_array);
    
    magma_free_cpu(aux_data.max_m);
    magma_free_cpu(aux_data.max_n);
    magma_free_cpu(aux_data.max_k);
    
    multimatmul_magma_grouped_aux_data_array.erase(aux_index);
}

torch::Tensor multi_row_sum_reduction(const torch::Tensor& input_matrix_tensor, const torch::Tensor& batch_size_per_network_tensor)
{

    magma_int_t *h_m, *h_n, *h_inc, *d_m, *d_n, *d_inc;
    float **h_A_array, **h_x_array, **h_y_array, **d_A_array, **d_x_array, **d_y_array;

    float alpha = MAGMA_S_MAKE(1.0f, 0.0f);
    float beta  = MAGMA_S_MAKE(0.0f, 0.0f);
    magma_int_t batchCount = batch_size_per_network_tensor.size(0);

    int total_batch_size = input_matrix_tensor.size(0);
    magma_int_t m = input_matrix_tensor.size(1);
    int max_batch_size = batch_size_per_network_tensor.max().data_ptr<int64_t>()[0];
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input_matrix_tensor.device());
    torch::Tensor x_tensor = torch::ones({max_batch_size}, options);
    torch::Tensor y_tensor = torch::zeros({batchCount, m}, options); // TODO: can also use empty?

    int64_t *batch_size_per_network = batch_size_per_network_tensor.data_ptr<int64_t>();
    float *input_matrix = input_matrix_tensor.data_ptr<float>();
    float *x = x_tensor.data_ptr<float>();
    float *y = y_tensor.data_ptr<float>();
    
    /* host allocs */
    magma_imalloc_cpu(&h_m, batchCount);
    magma_imalloc_cpu(&h_n, batchCount);
    magma_imalloc_cpu(&h_inc, batchCount);
    magma_malloc_cpu((void**)&h_A_array, batchCount * sizeof(float*));
    magma_malloc_cpu((void**)&h_x_array, batchCount * sizeof(float*));
    magma_malloc_cpu((void**)&h_y_array, batchCount * sizeof(float*));
    
    /* device allocs */
    magma_imalloc(&d_m, batchCount + 1);
    magma_imalloc(&d_n, batchCount + 1);
    magma_imalloc(&d_inc, batchCount + 1);
    magma_malloc((void**)&d_A_array, (batchCount + 1) * sizeof(float*));
    magma_malloc((void**)&d_x_array, (batchCount + 1) * sizeof(float*));
    magma_malloc((void**)&d_y_array, (batchCount + 1) * sizeof(float*));
    
    /* auxilary structure population */
    int A_offset = 0;
    int y_offset = 0;
    for (int i = 0; i < batchCount; ++i) {
        int n = batch_size_per_network[i];
        h_m[i] = m;
        h_n[i] = n;
        h_inc[i] = 1;
        h_A_array[i] = &input_matrix[A_offset];
        h_x_array[i] = x;
        h_y_array[i] = &y[y_offset];
        A_offset += n * m;
        y_offset += m;
    }
    
    /* host to device transfer */
    magma_setvector(batchCount, sizeof(magma_int_t), h_m, 1, d_m, 1, magma_queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_n, 1, d_n, 1, magma_queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_inc, 1, d_inc, 1, magma_queue);
    magma_setvector(batchCount, sizeof(float*), h_A_array, 1, d_A_array, 1, magma_queue);
    magma_setvector(batchCount, sizeof(float*), h_x_array, 1, d_x_array, 1, magma_queue);
    magma_setvector(batchCount, sizeof(float*), h_y_array, 1, d_y_array, 1, magma_queue);

    magma_int_t *d_ldda = d_m;

    magmablas_sgemv_vbatched(MagmaNoTrans,
        d_m,
        d_n,
        alpha,
        d_A_array,
        d_ldda,
        d_x_array,
        d_inc,
        beta,
        d_y_array,
        d_inc,
        batchCount,
        magma_queue
    );
    
    magma_free_cpu(h_m);
    magma_free_cpu(h_n);
    magma_free_cpu(h_inc);
    magma_free_cpu(h_A_array);
    magma_free_cpu(h_x_array);
    magma_free_cpu(h_y_array);

    magma_free(d_m);
    magma_free(d_n);
    magma_free(d_inc);
    magma_free(d_A_array);
    magma_free(d_x_array);
    magma_free(d_y_array);
    
    return y_tensor;
}

// hardcoded for backward pass
torch::Tensor multimatmul_A_transposed(const torch::Tensor& A_tensor, const torch::Tensor& B_tensor, const torch::Tensor& batch_size_per_network_tensor)
{

    magma_int_t m = B_tensor.size(1);
    magma_int_t n = A_tensor.size(1);
    magma_int_t batchCount = batch_size_per_network_tensor.size(0);

    torch::Tensor result_tensor = torch::zeros({batchCount, n, m}, A_tensor.options());

    float alpha = MAGMA_S_MAKE(1.0f, 0.f);
    float beta  = MAGMA_S_MAKE(0.0f,  0.0f);
    
    float* A = B_tensor.data_ptr<float>(); // already exchanged here
    float* B = A_tensor.data_ptr<float>();
    float* result = result_tensor.data_ptr<float>();
    
    int64_t *batch_size_per_network = batch_size_per_network_tensor.data_ptr<int64_t>();
    
    float **h_A_array = NULL;
    float **h_B_array = NULL;
    float **h_C_array = NULL;
    float **d_A_array = NULL;
    float **d_B_array = NULL;
    float **d_C_array = NULL;
        
    magma_int_t *h_M, *h_N, *h_K;
    magma_int_t *d_M, *d_N, *d_K; 
     
    magma_imalloc_cpu(&h_M, batchCount);
    magma_imalloc_cpu(&h_N, batchCount);
    magma_imalloc_cpu(&h_K, batchCount);
    magma_imalloc(&d_M, batchCount + 1);
    magma_imalloc(&d_N, batchCount + 1);
    magma_imalloc(&d_K, batchCount + 1);
    
    // pointer arrays
    magma_malloc_cpu((void**)&h_A_array, batchCount * sizeof(float*));
    magma_malloc_cpu((void**)&h_B_array, batchCount * sizeof(float*));
    magma_malloc_cpu((void**)&h_C_array, batchCount * sizeof(float*));
    magma_malloc((void**)&d_A_array, (batchCount + 1) * sizeof(float*));
    magma_malloc((void**)&d_B_array, (batchCount + 1) * sizeof(float*));
    magma_malloc((void**)&d_C_array, (batchCount + 1) * sizeof(float*));
    
    magma_int_t A_offset = 0;
    magma_int_t B_offset = 0;
    magma_int_t result_offset = 0;
    for (int i = 0; i < batchCount; ++i) {
        magma_int_t k = batch_size_per_network[i];
        
        h_M[i] = m;
        h_N[i] = n;
        h_K[i] = k;
        h_A_array[i] = &A[A_offset];
        h_B_array[i] = &B[B_offset];
        h_C_array[i] = &result[result_offset];

        A_offset += m * k;
        B_offset += k * n;
        result_offset += m * n;
    }
    
    magma_setvector(batchCount, sizeof(magma_int_t), h_M, 1, d_M, 1, magma_queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, magma_queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_K, 1, d_K, 1, magma_queue);

    magma_setvector(batchCount, sizeof(float*), h_A_array, 1, d_A_array, 1, magma_queue );
    magma_setvector(batchCount, sizeof(float*), h_B_array, 1, d_B_array, 1, magma_queue );
    magma_setvector(batchCount, sizeof(float*), h_C_array, 1, d_C_array, 1, magma_queue );
    
    magma_int_t *d_ldda = d_M;
    magma_int_t *d_lddb = d_N;
    magma_int_t *d_lddc = d_M;
    
    magmablas_sgemm_vbatched(MagmaNoTrans, MagmaTrans,
         d_M, d_N, d_K,
         alpha, d_A_array, d_ldda,
                d_B_array, d_lddb,
         beta,  d_C_array, d_lddc,
         batchCount,
         magma_queue);

    magma_free_cpu(h_M);
    magma_free_cpu(h_N);
    magma_free_cpu(h_K);
    magma_free(d_M);
    magma_free(d_N);
    magma_free(d_K);

    magma_free_cpu(h_A_array);
    magma_free_cpu(h_B_array);
    magma_free_cpu(h_C_array);
    magma_free(d_A_array);
    magma_free(d_B_array);
    magma_free(d_C_array);
    
    return result_tensor;
}
