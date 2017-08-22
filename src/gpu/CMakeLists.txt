### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

if(CUDA_FOUND)
  # caches variable, accessible by ccmake
  set(MCEA_CUDA_ARCH "" CACHE STRING "Target Compute Capability, e.g. '52' or '' = automatic (linux).")

  # Pass options to NVCC
  if(UNIX AND NOT MCEA_CUDA_ARCH)
    message(STATUS "Running automatic device architecture detection ...")
    make_directory(${CMAKE_BINARY_DIR}/tmp)
    execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/get_cuda_sm.sh 0
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tmp
      ERROR_FILE ${CMAKE_BINARY_DIR}/tmp/cmake.error.log
      OUTPUT_VARIABLE MCEA_CUDA_ARCH)
  endif()

  if(MCEA_CUDA_ARCH)
    message(STATUS "cuda arch: sm_" ${MCEA_CUDA_ARCH})
  else()
    message(WARNING "Could not determine CUDA architecture, please manually define MCEA_CUDA_ARCH (via ccmake or -D).")
    return()
  endif()

  set( CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/gpu)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  FILE(GLOB CONFIG_FILES "./config/*.cmake" )
  FOREACH(CFILE ${CONFIG_FILES})
    FILE(READ ${CFILE} CONFIG)
    STRING(REGEX REPLACE ";" "\\\\;" CONFIG "${CONFIG}")
    STRING(REGEX REPLACE "\n" ";" CONFIG "${CONFIG}")

    FOREACH(PARAM ${CONFIG})
      STRING(REPLACE " " ";" PARAM ${PARAM})
      LIST(GET PARAM 0 KEY)
      LIST(GET PARAM 1 VALUE)
      SET(${KEY} ${VALUE})
    ENDFOREACH()

    SET(CONFIG _g${GENERATIONS}_pw${POP_WIDTH}_p${PARAMS}_r${N_RAD}_t${THREADS}_vs${VADS_SCALE}_dt${DTLZ_NUM})
    SET(OUTFILE "\"${OUTFILE}${CONFIG}\"")

    set(
        CUDA_NVCC_FLAGS
        -arch=sm_${MCEA_CUDA_ARCH}
        --relocatable-device-code true
        -DGENERATIONS=${GENERATIONS}
        -DPOP_WIDTH=${POP_WIDTH}
        -DPARAMS=${PARAMS}
        -DN_RAD=${N_RAD}
        -DTHREADS=${THREADS}
        -DVADS_SCALE=${VADS_SCALE}
        -DOUTFILE=${OUTFILE}
        -DDTLZ_NUM=${DTLZ_NUM}
        )

    # For compilation ...
    # Specify target & source files to compile it from
    cuda_add_executable(
        mcea_sync${CONFIG}
        mcea_sync.cu
        util.cu
        dtlz.cu
        config.h
        error.h
        dtlz.cuh
        util.h
        )

    cuda_add_executable(
        mcea_async${CONFIG}
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
      mcea_sync${CONFIG}
      ${CUDA_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      rt
      )

    target_link_libraries(
      mcea_async${CONFIG}
      ${CUDA_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      rt
      )

  ENDFOREACH()
else()
  message(STATUS "CUDA deselected, toolchain not found on system")
endif()
