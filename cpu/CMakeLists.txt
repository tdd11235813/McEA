### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "-ffast-math -O3 -fopenmp -lm")

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

  # For compilation ...
  # Specify target & source files to compile it from
  add_executable(
      mcea_cpu${CONFIG}
      mcea.cpp
      util.cpp
      dtlz.cpp
      config.h
      dtlz.h
      util.h
      )

  target_compile_definitions(
      mcea_cpu${CONFIG}
      PRIVATE -DGENERATIONS=${GENERATIONS}
      PRIVATE -DPOP_WIDTH=${POP_WIDTH}
      PRIVATE -DPARAMS=${PARAMS}
      PRIVATE -DN_RAD=${N_RAD}
      PRIVATE -DTHREADS=${THREADS}
      PRIVATE -DVADS_SCALE=${VADS_SCALE}
      PRIVATE -DOUTFILE=${OUTFILE}
      PRIVATE -DDTLZ_NUM=${DTLZ_NUM}
      )

  TARGET_LINK_LIBRARIES(mcea_cpu${CONFIG}  ${OpenMP_CXX_LIBRARIES} ${PROJECT_LINK_LIBS})
ENDFOREACH()
