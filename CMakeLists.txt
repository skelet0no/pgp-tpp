cmake_minimum_required(VERSION 4.3)

set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CUDAToolkit_ROOT /opt/cuda)

project(pgp_tpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(info info.cu)
add_executable(hex_to_bin hex_to_bin.cpp)
add_executable(hw1 hw1/hw1.cpp)
add_executable(hw2 hw2/hw2.cpp)
add_executable(lab1_cuda lab1/lab1.cu)
add_executable(lab1_cpu lab1/lab1.cpp)
add_executable(lab1_cuda_test lab1/lab1_cuda_test.cu)
add_executable(lab1_cpu_vs_gpu lab1/lab1_cpu_vs_gpu.cu)
add_executable(lab2_cuda lab2/lab2.cu)
add_executable(lab2_cpu lab2/lab2.cpp)
add_executable(lab2_cuda_test lab2/lab2_cuda_test.cu)
add_executable(lab2_cpu_vs_gpu lab2/lab2_cpu_vs_gpu.cu)
add_executable(lab3_cuda lab3/lab3.cu)
add_executable(lab3_cpu lab3/lab3.cpp)
add_executable(lab3_cuda_test lab3/lab3_cuda_test.cu)
add_executable(lab3_cpu_vs_gpu lab3/lab3_cpu_vs_gpu.cu)