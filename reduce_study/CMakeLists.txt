cmake_minimum_required(VERSION 3.20)

project(reduce_study LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(reduce_study 02.reduce_sharedMem_no_divergence.cu)
set_target_properties(reduce_study PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
