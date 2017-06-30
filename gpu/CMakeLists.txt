### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

# Pass options to NVCC
execute_process(COMMAND ./get_cuda_sm.sh
                OUTPUT_VARIABLE CUDA_ARCH)

set( CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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

  SET(CONFIG _g${GENERATIONS}_pw${POP_WIDTH}_p${PARAMS}_r${NRAD}_t${THREADS}_vs${VADS_SCALE})
  SET(OUTFILE "\"${OUTFILE}${CONFIG}\"")

  set(
      CUDA_NVCC_FLAGS
      -arch=sm_${CUDA_ARCH} 
      --relocatable-device-code true
      -DGENERATIONS=${GENERATIONS}
      -DPOP_WIDTH=${POP_WIDTH}
      -DPARAMS=${PARAMS}
      -DN_RAD=${N_RAD}
      -DTHREADS=${THREADS}
      -DVADS_SCALE=${VADS_SCALE}
      -DOUTFILE=${OUTFILE}
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
      ${CUDA_curand_LIBRARY}
      )

    target_link_libraries(
      mcea_async${CONFIG}
      ${CUDA_curand_LIBRARY}
      )

ENDFOREACH()
