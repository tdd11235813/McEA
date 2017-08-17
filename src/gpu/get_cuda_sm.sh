#!/bin/bash
#
# Prints the compute capability of the first CUDA device installed
# on the system, or alternatively the device whose index is the
# first command-line argument

device_index=${1:-0}
timestamp=$(date +%s.%N)
nvcc_exec=$(which nvcc)
generated_source="/tmp/cuda-compute-version-helper-$$-$timestamp.cu"
generated_binary="/tmp/cuda-compute-version-helper-$$-$timestamp"
# create a 'here document' that is code we compile and use to probe the card
echo " 
#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
	cudaDeviceProp prop;
	cudaError_t status;
	int device_count;
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) {
		fprintf(stderr,\"cudaGetDeviceCount() failed: %s\n\", cudaGetErrorString(status));
		return -1;
	}
	if (${device_index} >= device_count) {
		fprintf(stderr, \"Specified device index %d exceeds the maximum (the device count on this system is %d)\n\", ${device_index}, device_count);
		return -1;
	}
	status = cudaGetDeviceProperties(&prop, ${device_index});
	if (status != cudaSuccess) {
		fprintf(stderr,\"cudaGetDeviceProperties() for device ${device_index} failed: %s\n\", cudaGetErrorString(status));
		return -1;
	}
	int v = prop.major * 10 + prop.minor;
	printf(\"%d\", v);
}" > "$generated_source"

$nvcc_exec -o $generated_binary $generated_source 2> /dev/null
# echo "$source_code" | $gcc_binary -x c++ -I"$CUDA_INCLUDE_DIRS" -o "$generated_binary" - -x none "$CUDA_CUDART_LIBRARY"

# probe the card and cleanup

$generated_binary
rm $generated_source
rm $generated_binary
