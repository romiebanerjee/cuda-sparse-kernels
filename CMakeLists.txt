cmake_minimum_required(VERSION 3.10)
project(CUDA_SPMM)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

# Enable CUDA separable compilation
set(CUDA_SEPARABLE_COMPILATION ON)

# Add executable
add_executable(spmm_example examples/basic_example.cu src/kernel.cu src/wrapper.cpp)

# Link CUDA libraries
target_link_libraries(spmm_example PRIVATE CUDA::cudart)

# Add tests
enable_testing()
add_executable(test_spmm tests/test_spmm.cpp src/kernel.cu src/wrapper.cpp src/utils.cpp)
target_link_libraries(test_spmm PRIVATE CUDA::cudart)
add_test(NAME test_spmm COMMAND test_spmm)