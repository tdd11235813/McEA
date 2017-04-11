#include "cuda.h"

// own header files
#include "error.h"
#include "util.h"
#include "dtlz.cuh"

#define POP_WIDTH 10
#define POP_SIZE ((POP_WIDTH * (POP_WIDTH + 1)) / 2)
#define PARAMS 50
#define OBJS 3


__global__ void mcea( float *population, float *objectives, float *utopia_vec ) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if( idx < POP_SIZE ) {
    testObjSum( population+idx, objectives+idx, PARAMS, OBJS );
  }

  return;
}

int main() {
  // allocate memory
  float population_h[POP_SIZE][PARAMS];
  float objectives_h[POP_SIZE][OBJS];
  float utopia_vec_h[OBJS];
  float *population_d;
  float *objectives_d;
  float *utopia_vec_d;

  ERR( cudaMalloc( (void**)&population_d, POP_SIZE * PARAMS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&objectives_d, POP_SIZE * OBJS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&utopia_vec_d, OBJS * sizeof(float) ) );

  // create random population
  srand( time( NULL ) );
  for (size_t i = 0; i < POP_SIZE; i++) {
    for (size_t j = 0; j < PARAMS; j++) {
      population_h[i][j] = randomFloat();
    }
  }

  // copy data to GPU
  ERR( cudaMemcpy( population_d, population_h, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyHostToDevice ) );

  // start the kernel
  mcea<<<1, POP_SIZE>>>( population_d, objectives_d, utopia_vec_d );

  // copy data from GPU
  ERR( cudaMemcpy( population_h, population_d, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyDeviceToHost ) );
  ERR( cudaMemcpy( objectives_h, objectives_d, POP_SIZE * OBJS * sizeof(float), cudaMemcpyDeviceToHost ) );
  ERR( cudaMemcpy( utopia_vec_h, utopia_vec_d, OBJS * sizeof(float), cudaMemcpyDeviceToHost ) );

  // print some solutions
  printVector( population_h[0], PARAMS );
  printVector( objectives_h[0], OBJS );

  // free resources
  ERR( cudaFree( population_d ) );
  ERR( cudaFree( objectives_d ) );
  ERR( cudaFree( utopia_vec_d ) );
}
