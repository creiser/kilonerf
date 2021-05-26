
#include "multimatmul.cuh"
#include "fourier_features.cuh"
#include "generate_inputs.cuh"
#include "global_to_local.cuh"
#include "network_eval.cuh"
#include "integrate.cuh"
#include "reorder.cuh"
#include "render_to_screen.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init_stream_pool", &init_stream_pool, "");
    m.def("destroy_stream_pool", &destroy_stream_pool, "");

    m.def("init_magma", &init_magma, "");
    m.def("multimatmul_magma_grouped_static", &multimatmul_magma_grouped_static, "");
    m.def("multimatmul_magma_grouped_static_without_bias", &multimatmul_magma_grouped_static_without_bias, "");
    m.def("multimatmul_magma_grouped_static_without_bias_transposed_weights", &multimatmul_magma_grouped_static_without_bias_transposed_weights, "");

    m.def("init_multimatmul_magma_grouped", &init_multimatmul_magma_grouped, "");
    m.def("deinit_multimatmul_magma_grouped", &deinit_multimatmul_magma_grouped, "");

    m.def("multi_row_sum_reduction", &multi_row_sum_reduction, "");
    m.def("multimatmul_A_transposed", &multimatmul_A_transposed, "");

    m.def("gather_int32", &gather_int32, "");
    m.def("scatter_int32_float4", &scatter_int32_float4, "");
    m.def("sort_by_key_int16_int64", &sort_by_key_int16_int64, "");
    m.def("sort_by_key_int16_int32", &sort_by_key_int16_int32, "");

    m.def("get_rays_d", &get_rays_d, "");
    m.def("generate_query_indices_on_ray", &generate_query_indices_on_ray, "");
    m.def("global_to_local", &global_to_local, "");
    m.def("compute_fourier_features", &compute_fourier_features, "");
    m.def("network_eval_query_index", &network_eval_query_index, "");
    m.def("integrate", &integrate, "");
    m.def("replace_transparency_by_background_color", &replace_transparency_by_background_color, "");
    m.def("render_to_screen", &render_to_screen, "");
}