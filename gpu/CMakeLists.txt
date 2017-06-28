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
    -DGENERATIONS=100
    )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    mcea_sync_g100
    mcea_sync.cu
    util.cu
    dtlz.cu
    config.h
    error.h
    dtlz.cuh
    util.h
    )

cuda_add_executable(
    mcea_async_g100
    mcea_async.cu
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
    mcea_sync_g100
    ${CUDA_curand_LIBRARY}
    )

target_link_libraries(
    mcea_async_g100
    ${CUDA_curand_LIBRARY}
    )

set(
    CUDA_NVCC_FLAGS
    -DGENERATIONS=10
    )

cuda_add_executable(
    mcea_sync_g10
    mcea_sync.cu
    util.cu
    dtlz.cu
    config.h
    error.h
    dtlz.cuh
    util.h
    )

cuda_add_executable(
    mcea_async_g10
    mcea_async.cu
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
    mcea_sync_g10
    ${CUDA_curand_LIBRARY}
    )

target_link_libraries(
    mcea_async_g10
    ${CUDA_curand_LIBRARY}
    )
set(
    CUDA_NVCC_FLAGS
    -DGENERATIONS=1000
    )

cuda_add_executable(
    mcea_sync_g1000
    mcea_sync.cu
    util.cu
    dtlz.cu
    config.h
    error.h
    dtlz.cuh
    util.h
    )

cuda_add_executable(
    mcea_async_g1000
    mcea_async.cu
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
    mcea_sync_g1000
    ${CUDA_curand_LIBRARY}
    )

target_link_libraries(
    mcea_async_g1000
    ${CUDA_curand_LIBRARY}
    )
