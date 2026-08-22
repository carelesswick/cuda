cmake_minimum_required(VERSION 3.20)

project(reduce_study LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(reduce_study 03.reduce_no_bank_conflict.cu)

target_compile_options(reduce_study PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G;-O0;-lineinfo>
)