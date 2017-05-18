### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

# Pass options to NVCC
execute_process(COMMAND ./get_cuda_sm.sh
                OUTPUT_VARIABLE CUDA_ARCH)
set( CUDA_SEPARABLE_COMPILATION ON)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_${CUDA_ARCH} --relocatable-device-code true
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    mcea
    mcea.cu
    util.cu
    dtlz.cu
    config.h
    error.h
    dtlz.cuh
    util.h
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    mcea
    ${CUDA_curand_LIBRARY}
    )
