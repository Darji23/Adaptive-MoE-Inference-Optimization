cmake_minimum_required(VERSION 3.18)
project(moe_kv_kernels CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 90)  # H100 = SM90
find_package(CUDA REQUIRED)

add_library(moe_kv_kernels SHARED
    kv_eviction_policy.cu
    moe_router_profiler.cu
    ssm_state_monitor.cu
)

target_compile_options(moe_kv_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --ptxas-options=-v
        -gencode arch=compute_90,code=sm_90
    >
)
