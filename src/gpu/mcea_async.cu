/*! \file mcea_async.cu
  Main algorithm. Does all the memory management and starts the kernels.
*/
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <string>
#include <vector>

// own header files
#include "../util/output.cuh"
#include "../util/neighbor.cuh"
#include "../util/random.cuh"
#include "../util/dtlz.h"
#include "../util/weighting.cuh"
#include "../util/error.h"
#include "config.h"

/*! \brief McEA kernel

  This kernel runs the whole algorithm. All data structures have to be set up prior to this.
  It uses the population and performs GENERATIONS generations, consisting of pairing, crossover, mutation, evaluation, and selection on it.
  At the end the population contains the optimized individuals.

  \param[in,out] population an array containing all parameters of the whole population.
  \param[out] objectives an array containing all objective values 
  \param[in] rng_state the initialized state of the PRNG to use
*/
__global__ void mcea( float *population, float *objectives, curandStatePhilox4_32_10_t *rng_state ) {
  __shared__ float offspring[PARAMS * BLOCKSIZE];
  __shared__ float offspring_fit[OBJS * BLOCKSIZE];
  __shared__ float weights[OBJS * BLOCKSIZE];
  curandStatePhilox4_32_10_t rng_local;
  float4_union randn_neigh_1, randn_neigh_2, randn_xover_point;
  int4_union randn_mut_count;
  double fit_parent;

  // global indices
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);
  // blockwise indices
  int block_idx = (blockDim.x * threadIdx.y + threadIdx.x);

  // init RNG, fitness and weights
  if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
    rng_local = *(rng_state + idx);
    // ### evaluation ###
    dtlz( population+idx, objectives+idx, PARAMS, OBJS, POP_SIZE );
    calc_weights(x, y, weights + block_idx, BLOCKSIZE);
    fit_parent =  weighted_fitness( objectives + idx, weights + block_idx, POP_SIZE );
  }

  // main loop
#if STOPTYPE == GENERATIONS
  // stop after number of generations
  for (size_t g = 0; g < STOPVALUE; g++) {
#elif STOPTYPE == TIME
    if(idx == 0)
      printf( "STOPTYPE: TIME is not possible for async computation. doing just one generation.\n" );
    {
      int g = 0;
#else
    if(idx == 0)
      printf( "no valid STOPTYPE. doing just one generation.\n" );
    {
      int g = 0;
#endif

    if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {

      // ### generate random numbers ###
      // pre-generate for 4 generations
      if(g % 4 == 0) {
        randn_neigh_1.vec = curand_uniform4( &rng_local );
        randn_neigh_2.vec = curand_uniform4( &rng_local );
        randn_xover_point.vec = curand_uniform4( &rng_local );
        randn_mut_count.vec = curand_poisson4( &rng_local, LAMBDA );

      }

      // ### pairing ###
      // random neighbors
      int neighbor_1 = get_neighbor( x, y, trans_uniform_int( randn_neigh_1.arr[g%4], N_WIDTH * N_WIDTH ) );
      int neighbor_2 = get_neighbor( x, y, trans_uniform_int( randn_neigh_2.arr[g%4], N_WIDTH * N_WIDTH ) );

      // compare neighbors
      double fit_1 =  weighted_fitness( objectives + neighbor_1, weights + block_idx, POP_SIZE );
      double fit_2 =  weighted_fitness( objectives + neighbor_2, weights + block_idx, POP_SIZE );
      int neighbor_sel = (fit_1 < fit_2)? neighbor_1 : neighbor_2;

      if( idx == 0 && VERBOSE )
        printf("x: %d, y: %d, n1: %3d(%.3f), n2: %3d(%.3f), sel: %3d\n", x, y, neighbor_1, fit_1, neighbor_2, fit_2, neighbor_sel);

      if( idx == 0 && VERBOSE ) {
        printf( "original: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", population[idx + i * POP_SIZE] );
        printf( "\n" );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", objectives[i + idx * POP_SIZE] );
        printf( "\n" );
      }

      // ### crossover ###
      // == one-point crossover
      int x_over_point = trans_uniform_int( randn_xover_point.arr[g%4], PARAMS );
      if( idx == 0 && VERBOSE )
        printf( "xover: %d\n", x_over_point );

      for (size_t i = 0; i < PARAMS; i++)
        offspring[block_idx + BLOCKSIZE * i] = (i<x_over_point) ? population[idx + i * POP_SIZE] : population[neighbor_sel + i * POP_SIZE];

      if( idx == 0 && VERBOSE ) {
        printf( "crossover: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", offspring[block_idx + BLOCKSIZE * i] );
        printf( "\n" );
      }
      // ### mutation ###
      // == uniform mutation
      int num_mutations = randn_mut_count.arr[g%4];
      if( idx == 0 && VERBOSE )
        printf( "mut: %d\n", num_mutations );

      for (size_t i = 0; i < num_mutations; i++) {
        int mut_location = trans_uniform_int( curand_uniform(&rng_local), PARAMS );
        offspring[block_idx + BLOCKSIZE * mut_location] = curand_uniform( &rng_local );
      }

      if( idx == 0 && VERBOSE ) {
        printf( "mutated: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", offspring[block_idx + BLOCKSIZE * i] );
        printf( "\n" );
      }

      // ### selection ###
      // == select if better

      // evaluate the offspring
      dtlz( offspring + block_idx , offspring_fit + block_idx, PARAMS, OBJS, BLOCKSIZE );

      if( idx == 0 && VERBOSE ) {
        printf( "offspring fit: " );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", offspring_fit[block_idx + BLOCKSIZE * i] );
        printf( "\n" );
      }

      // compare and copy
      fit_2 =  weighted_fitness( offspring_fit + block_idx, weights + block_idx, BLOCKSIZE );

      if( idx == 0 && VERBOSE ) {
        printf( "offspring weight: %.5lf\n", fit_2 );
      }

      if(fit_2 < fit_parent) {
        for (size_t i = 0; i < PARAMS; i++)
          population[idx + i * POP_SIZE] = offspring[block_idx + BLOCKSIZE * i];
        for (size_t i = 0; i < OBJS; i++)
          objectives[idx + i * POP_SIZE] = offspring_fit[block_idx + BLOCKSIZE * i];
        fit_parent = fit_2;
      }

      if( idx == 0 && VERBOSE ) {
        printf( "new ind: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", population[idx + i * POP_SIZE] );
        printf( "\n" );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", objectives[idx + i * POP_SIZE] );
        printf( "\n" );
      }
    }

    // sync the block after every generation
    __syncthreads();
  }

  return;
}

/*! \brief main function

  Classic main function. It allocates all memory, generates the population, starts the kernel and collects the results.
  All parameter changes are made via the \#define statements
*/
int main(int argc, char *argv[]) {

  // get the output folder the run number and type
  string folder = "";
  string run = "0";
  if(argc > 1) {
    folder = argv[1];
    run = argv[2];
  }

  run = string("async_") + run;

  // allocate memory
  float *population_h = (float *)malloc( POP_SIZE * PARAMS * sizeof(float) );
  float *objectives_h = (float *)malloc( POP_SIZE * OBJS * sizeof(float) );
  float *population_d;
  float *objectives_d;

  curandStatePhilox4_32_10_t *d_state;
  ERR( cudaMalloc( &d_state, POP_SIZE * sizeof(curandStatePhilox4_32_10_t) ) );
  ERR( cudaMalloc( (void**)&population_d, POP_SIZE * PARAMS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&objectives_d, POP_SIZE * OBJS * sizeof(float) ) );

  // setup random generator
  unsigned long seed = clock();
  rand_init<<<POP_SIZE / 1024 + 1, 1024>>>( d_state, seed );

  // create random population
  srand( time( NULL ) );
  for (size_t i = 0; i < POP_SIZE; i++) {
    for (size_t j = 0; j < PARAMS; j++) {
      population_h[i * PARAMS + j] = randomFloat();
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
  dim3 dimBlock(BLOCKDIM, BLOCKDIM);
  dim3 dimGrid(ceil((POP_WIDTH + 1) / (float)BLOCKDIM) , ceil(POP_WIDTH / (float)BLOCKDIM));
  mcea<<<dimGrid, dimBlock>>>( population_d, objectives_d, d_state );

  // get stop time, and display the timing results
  ERR( cudaEventRecord( stop, 0 ) );
  ERR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  ERR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf( "duration:  %f ms\n", elapsedTime );

  // copy data from GPU
  ERR( cudaMemcpy( population_h, population_d, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyDeviceToHost ) );
  ERR( cudaMemcpy( objectives_h, objectives_d, POP_SIZE * OBJS * sizeof(float), cudaMemcpyDeviceToHost ) );

  // write the results to file
  write_objectives( objectives_h, folder, run );
  write_info( elapsedTime, folder, run );

  // free resources
  free( population_h );
  free( objectives_h );
  ERR( cudaEventDestroy( start ) );
  ERR( cudaEventDestroy( stop ) );

  ERR( cudaFree( population_d ) );
  ERR( cudaFree( objectives_d ) );
  ERR( cudaFree( d_state ) );
}
