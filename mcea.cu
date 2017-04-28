/*! \file mcea.cu
  Main algorithm. Does all the memory management and starts the kernels.
*/
#include "cuda.h"

// own header files
#include "error.h"
#include "util.h"
#include "dtlz.cuh"

#define POP_WIDTH 10
#define POP_SIZE ((POP_WIDTH * (POP_WIDTH + 1)) / 2)
#define PARAMS 100
#define OBJS 5

/*! \brief main kernel

  This kernel runs the whole algorithm. All data structures have to be set up for this.
  TODO: implement algorithm
  \param population an array containing all parameters of the whole population.
  \param objectives an array containing all objective values (there will be written some new ones)
  \param utopia_vec a vector containing the best values for each single objective
*/
__global__ void mcea( float *population, float *objectives, float *utopia_vec ) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if( idx < POP_SIZE ) {
    dtlz1( population+idx, objectives+idx, PARAMS, OBJS );
  }

  return;
}

/*! \brief main function

  Classic main function. It allocates all memory, generates the population, starts the kernel and collects the results.
  All parameters changes are made via the #define statements
*/
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
      //population_h[i][j] = randomFloat();
      population_h[i][j] = 1.0 / (j+1);
    }
  }

  // copy data to GPU
  ERR( cudaMemcpy( population_d, population_h, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyHostToDevice ) );

  // capture the start time
  cudaEvent_t     start, stop;
  ERR( cudaEventCreate( &start ) );
  ERR( cudaEventCreate( &stop ) );
  ERR( cudaEventRecord( start, 0 ) );

  // start the kernel
  mcea<<<1, POP_SIZE>>>( population_d, objectives_d, utopia_vec_d );

  // get stop time, and display the timing results
  ERR( cudaEventRecord( stop, 0 ) );
  ERR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  ERR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf( "Time to generate:  %f ms\n", elapsedTime );

  // copy data from GPU
  ERR( cudaMemcpy( population_h, population_d, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyDeviceToHost ) );
  ERR( cudaMemcpy( objectives_h, objectives_d, POP_SIZE * OBJS * sizeof(float), cudaMemcpyDeviceToHost ) );
  ERR( cudaMemcpy( utopia_vec_h, utopia_vec_d, OBJS * sizeof(float), cudaMemcpyDeviceToHost ) );

  ERR( cudaEventDestroy( start ) );
  ERR( cudaEventDestroy( stop ) );

  // print some solutions
  printVector( population_h[0], PARAMS );
  printVector( objectives_h[0], OBJS );

  // free resources
  ERR( cudaFree( population_d ) );
  ERR( cudaFree( objectives_d ) );
  ERR( cudaFree( utopia_vec_d ) );
}
