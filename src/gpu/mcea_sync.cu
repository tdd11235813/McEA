/*! \file mcea_sync.cu
  Main algorithm. Does all the memory management and starts the kernels.
*/
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

// own header files
#include "../util/output.cuh"
#include "../util/random.cuh"
#include "../util/neighbor.cuh"
#include "../util/dtlz.h"
#include "../util/weighting.cuh"
#include "../util/error.h"
#include "config.h"

/*! \brief fitness kernel

   This kernel calculates the initial fitness of all randomly generated individuals.

  \param[in, out] population an array containing all parameters of the whole population.
  \param[in, out] objectives an array containing all objective (= fitness) values 
*/
__global__ void calc_fitness( float *population, float *objectives ) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);

  if( x < POP_WIDTH + 1 && y < POP_WIDTH )
    dtlz( population+idx, objectives+idx, PARAMS, OBJS, POP_SIZE );
}

/*! \brief McEA kernel

  This kernel runs the whole algorithm. All data structures have to be set up prior to this and the
  fitness must be calculated aleady (see: void calc_fitness() ).
  It uses the population and performs GENERATIONS generations, consisting of pairing, crossover, mutation, evaluation, and selection on it.
  At the end the population contains the optimized individuals.

  \param[in] population_in an array containing all parameters of the whole population.
  \param[in] objectives_in an array containing all objective values (there will be written some new ones)
  \param[out] population_out an array where all the parameters of the new population are written
  \param[out] objectives_out an array where all the objective values of the new population are written
  \param[in] rng_state the initialized state of the PRNG to use
*/
__global__ void mcea( float *population_in, float *objectives_in, float *population_out, float *objectives_out, curandStatePhilox4_32_10_t *rng_state ) {
  __shared__ float offspring[PARAMS * BLOCKSIZE];
  __shared__ float offspring_fit[OBJS * BLOCKSIZE];
  __shared__ float weights[OBJS * BLOCKSIZE];
  curandStatePhilox4_32_10_t rng_local;

  // global indices
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);
  // blockwise indices
  int block_idx = (blockDim.x * threadIdx.y + threadIdx.x);

  if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
    // preparation
    rng_local = *(rng_state + idx);
    calc_weights(x, y, weights + block_idx, BLOCKSIZE);

    // ### pairing ###
    // random neighbors
    int neighbor_1 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );
    int neighbor_2 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );

    // compare neighbors
    double fit_1 =  weighted_fitness( objectives_in + neighbor_1, weights + block_idx, POP_SIZE);
    double fit_2 =  weighted_fitness( objectives_in + neighbor_2, weights + block_idx, POP_SIZE);
    int neighbor_sel = (fit_1 < fit_2)? neighbor_1 : neighbor_2;

    if( idx == 0 && VERBOSE )
      printf("x: %d, y: %d, n1: %3d(%.3f), n2: %3d(%.3f), sel: %3d\n", x, y, neighbor_1, fit_1, neighbor_2, fit_2, neighbor_sel);

    if( idx == 0 && VERBOSE ) {
      printf( "original: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", population_in[idx + i * POP_SIZE] );
      printf( "\n" );
      for (size_t i = 0; i < OBJS; i++)
        printf( "%.2f, ", objectives_in[idx + i * POP_SIZE] );
      printf( "\n" );
    }
    // ### crossover ###
    // == one-point crossover
    int x_over_point = rnd_uniform_int( &rng_local, PARAMS );
    if( idx == 0 && VERBOSE )
      printf( "xover: %d\n", x_over_point );

    for (size_t i = 0; i < PARAMS; i++)
      offspring[block_idx + BLOCKSIZE * i] = (i<x_over_point) ? population_in[idx + i * POP_SIZE] : population_in[neighbor_sel + i * POP_SIZE];

    if( idx == 0 && VERBOSE ) {
      printf( "crossover: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", offspring[block_idx + BLOCKSIZE * i] );
      printf( "\n" );
    }
    // ### mutation ###
    // == uniform mutation
    int num_mutations = curand_poisson( &rng_local, LAMBDA );
    if( idx == 0 && VERBOSE )
      printf( "mut: %d\n", num_mutations );

    for (size_t i = 0; i < num_mutations; i++) {
      int mut_location = rnd_uniform_int( &rng_local, PARAMS );
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
    dtlz( offspring + block_idx, offspring_fit + block_idx, PARAMS, OBJS, BLOCKSIZE );

    if( idx == 0 && VERBOSE ) {
      printf( "offspring fit: " );
      for (size_t i = 0; i < OBJS; i++)
        printf( "%.2f, ", offspring_fit[block_idx + BLOCKSIZE * i] );
      printf( "\n" );
    }

    // compare and copy
    fit_1 =  weighted_fitness( objectives_in + idx, weights + block_idx, POP_SIZE );
    fit_2 =  weighted_fitness( offspring_fit + block_idx, weights + block_idx, BLOCKSIZE );

    if( idx == 0 && VERBOSE )
      printf( "offspring weight: %.5lf\n", fit_2 );

    if(fit_2 < fit_1) {
      for (size_t i = 0; i < PARAMS; i++)
        population_out[idx + i * POP_SIZE] = offspring[block_idx + BLOCKSIZE * i];
      for (size_t i = 0; i < OBJS; i++)
        objectives_out[idx + i * POP_SIZE] = offspring_fit[block_idx + BLOCKSIZE * i];
    }
    else {
      for (size_t i = 0; i < PARAMS; i++)
        population_out[idx + i * POP_SIZE] = population_in[idx + i * POP_SIZE];
      for (size_t i = 0; i < OBJS; i++)
        objectives_out[idx + i * POP_SIZE] = objectives_in[idx + i * POP_SIZE];
    }


    if( idx == 0 && VERBOSE ) {
      printf( "new ind: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", population_out[idx + i * POP_SIZE] );
      printf( "\n" );
      for (size_t i = 0; i < OBJS; i++)
        printf( "%.2f, ", objectives_out[idx + i * POP_SIZE] );
      printf( "\n" );
    }

    *(rng_state + idx) = rng_local;
  }

  return;
}

/*! \brief main function

  Classic main function. It allocates all memory, generates the population, starts the kernel and collects the results.
  All parameter changes are made via the \#define statements
*/
int main( int argc, char *argv[] ) {

  // get the output folder the run number and type
  string folder = "";
  string run = "0";
  if(argc > 1) {
    folder = argv[1];
    run = argv[2];
  }

  run = string("sync_") + run;

  // allocate memory
  float *population_h = (float *)malloc( POP_SIZE * PARAMS * sizeof(float) );
  float *objectives_h = (float *)malloc( POP_SIZE * OBJS * sizeof(float) );
  float *population1_d;
  float *objectives1_d;
  float *population2_d;
  float *objectives2_d;

  curandStatePhilox4_32_10_t *d_state;
  ERR( cudaMalloc( &d_state, POP_SIZE * sizeof(curandStatePhilox4_32_10_t) ) );
  ERR( cudaMalloc( (void**)&population1_d, POP_SIZE * PARAMS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&objectives1_d, POP_SIZE * OBJS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&population2_d, POP_SIZE * PARAMS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&objectives2_d, POP_SIZE * OBJS * sizeof(float) ) );

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
  ERR( cudaMemcpy( population1_d, population_h, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyHostToDevice ) );

  // capture the start time
  cudaEvent_t     start, stop;
  ERR( cudaEventCreate( &start ) );
  ERR( cudaEventCreate( &stop ) );
  ERR( cudaEventRecord( start, 0 ) );

  // start the kernel
  dim3 dimBlock(BLOCKDIM, BLOCKDIM);
  dim3 dimGrid(ceil((POP_WIDTH + 1) / (float)BLOCKDIM) , ceil(POP_WIDTH / (float)BLOCKDIM));
  calc_fitness<<<dimGrid, dimBlock>>>( population1_d, objectives1_d );
  for (int i = 0; i < GENERATIONS; i++) {
    mcea<<<dimGrid, dimBlock>>>( population1_d, objectives1_d, population2_d, objectives2_d, d_state );

    // switch buffers
    float *tmp = population1_d;
    population1_d = population2_d;
    population2_d = tmp;

    tmp = objectives1_d;
    objectives1_d = objectives2_d;
    objectives2_d = tmp;
  }

  // get stop time, and display the timing results
  ERR( cudaEventRecord( stop, 0 ) );
  ERR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  ERR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf( "duration:  %f ms\n", elapsedTime );

  // copy data from GPU
  ERR( cudaMemcpy( population_h, population1_d, POP_SIZE * PARAMS * sizeof(float), cudaMemcpyDeviceToHost ) );
  ERR( cudaMemcpy( objectives_h, objectives1_d, POP_SIZE * OBJS * sizeof(float), cudaMemcpyDeviceToHost ) );

  // write the results to file
  write_objectives( objectives_h, folder, run);
  write_info( elapsedTime, folder, run);

  // free resources
  free( population_h );
  free( objectives_h );
  ERR( cudaEventDestroy( start ) );
  ERR( cudaEventDestroy( stop ) );

  ERR( cudaFree( population1_d ) );
  ERR( cudaFree( objectives1_d ) );
  ERR( cudaFree( population2_d ) );
  ERR( cudaFree( objectives2_d ) );
}
