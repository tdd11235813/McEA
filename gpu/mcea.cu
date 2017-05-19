/*! \file mcea.cu
  Main algorithm. Does all the memory management and starts the kernels.
*/
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

// own header files
#include "error.h"
#include "util.h"
#include "dtlz.cuh"
#include "config.h"

/*! \brief neighbor calculation

  For a given neighbor index this calculates the neighbors global position.
  The neighbor index is a number representing the position of the neighbor in the neighborhood of the individual.
  It is organized rowwise starting at the top-left individual in the neighborhood (index = 0). When there is an
  overflow in any of the directions, the global index will be wrapped around to the other side of the population.

  \param[in] x the x position of the original individual
  \param[in] y the y position of the original individual
  \param[in] neighbor_index the position of the neighbor relative to the original. Allowed values depend on the population size.

  \return the global index of the neighbor
*/
__device__ int get_neighbor(int x, int y, int neighbor_index) {
  // 2D indices
  int n_x = (x + neighbor_index % N_WIDTH - N_RAD + POP_WIDTH + 1) % (POP_WIDTH + 1);
  int n_y = (y + neighbor_index / N_WIDTH - N_RAD + POP_WIDTH) % POP_WIDTH;

  // global index
  return n_x + n_y * (POP_WIDTH + 1);
}

/*! \brief init PRNG

Initializes one pseudo random number generator for each thread.
The same seed is used, but every thread uses a different sequence.

\param[out] state the data structure for the PRNG state
\param[in] seed the seed for initialization

\return nothing
*/
__global__ void rand_init( curandStatePhilox4_32_10_t  *state, unsigned long seed ) {
  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  curand_init(seed, idx, 0, &state[idx]);
}

/* \brief generates a random uniform int

  Draws from a uniform distribution in [0, 1] and converts it to an integer in the range [0, values-1].

  \param state[in] the PRNG state to use
  \param values[in] the number of possible values for the uniform distribution

  \return an integer in the specified range
*/
__device__ int rnd_uniform_int( curandStatePhilox4_32_10_t  *state, int values ) {

    return (int)truncf( curand_uniform( state ) * ( values - 0.000001) );
}

/* \brief calculates the weighted fitness

Takes the objective values of the individual at idx and calculates its fitness.
The specific weights for the individual at location x,y in the population are used for weighting.
! This only works for 3 objectives for now !
TODO: for real world problems use the weighted tchebychev method (use utopia vector)

\param[in] objectives pointer to the objective values of the individual
\param[in] x the x location of the weighting basis (does not have to be the same ind the objectives are from)
\param[in] y the y location of the weighting basis (does not have to be the same ind the objectives are from)

\return the weighted fitness value
*/
__device__ float weighted_fitness( float *objectives, int x, int y ) {
  // this decides if the individual is on the mirrored side of the population
  // and gives the correction factor for the weights
  int mirror = (x + y < POP_WIDTH)? false : true;

  float offset =  (mirror) ? 0.25 : 0.0;
  int _x  =  (mirror) ? POP_WIDTH - y - 1 : x;
  int _y  =  (mirror) ? POP_WIDTH - x : y;

  // calculate the fitness
  return \
      objectives[0] * (1 - (_x+offset)/(POP_WIDTH-0.5) - (_y+offset)/(POP_WIDTH-0.5)) \
    + objectives[1] * (_x+offset)/(POP_WIDTH-0.5) \
    + objectives[2] * (_y+offset)/(POP_WIDTH-0.5);

}

/*! \brief McEA kernel

  This kernel runs the whole algorithm. All data structures have to be set up prior to this.
  It uses the population and performs GENERATIONS generations, consisting of pairing, crossover, mutation, evaluation, and selection on it.
  At the end the population contains the optimized individuals.

  \param[in,out] population an array containing all parameters of the whole population.
  \param[in,out] objectives an array containing all objective values (there will be written some new ones)
  \param[in] rng_state the initialized state of the PRNG to use
*/
__global__ void mcea( float *population, float *objectives, curandStatePhilox4_32_10_t *rng_state ) {
  __shared__ float offspring[PARAMS * BLOCKDIM * BLOCKDIM];
  __shared__ float offspring_fit[OBJS * BLOCKDIM * BLOCKDIM];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int block_idx = (blockDim.x * threadIdx.y + threadIdx.x);
  int idx = x + y * (POP_WIDTH + 1);
  curandStatePhilox4_32_10_t rng_local;

  if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
    rng_local = *(rng_state + idx);
    // ### evaluation ###
    dtlz1( population+idx*PARAMS, objectives+idx*OBJS, PARAMS, OBJS );
  }

  // main loop
  for (size_t g = 0; g < GENERATIONS; g++) {

    if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
      // ### pairing ###
      // random neighbors
      int neighbor_1 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );
      int neighbor_2 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );

      // compare neighbors
      float fit_1 = weighted_fitness( objectives + neighbor_1 * OBJS, x, y );
      float fit_2 = weighted_fitness( objectives + neighbor_2 * OBJS, x, y );
      int neighbor_sel = (fit_1 < fit_2)? neighbor_1 : neighbor_2;

      if( idx == 0 && VERBOSE )
        printf("x: %d, y: %d, n1: %3d(%.3f), n2: %3d(%.3f), sel: %3d\n", x, y, neighbor_1, fit_1, neighbor_2, fit_2, neighbor_sel);

      if( idx == 0 && VERBOSE ) {
        printf( "original: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", population[i + idx * PARAMS] );
        printf( "\n" );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", objectives[i + idx * OBJS] );
        printf( "\n" );
      }
      // ### crossover ###
      // == one-point crossover
      int x_over_point = rnd_uniform_int( &rng_local, PARAMS );
      if( idx == 0 && VERBOSE )
        printf( "xover: %d\n", x_over_point );

      for (size_t i = 0; i < PARAMS; i++)
        offspring[block_idx * PARAMS + i] = (i<x_over_point) ? population[i + idx * PARAMS] : population[i + neighbor_sel * PARAMS];

      if( idx == 0 && VERBOSE ) {
        printf( "crossover: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", offspring[block_idx * PARAMS + i] );
        printf( "\n" );
      }
      // ### mutation ###
      // == uniform mutation
      int num_mutations = curand_poisson( &rng_local, LAMBDA );
      if( idx == 0 && VERBOSE )
        printf( "mut: %d\n", num_mutations );

      for (size_t i = 0; i < num_mutations; i++) {
        int mut_location = rnd_uniform_int( &rng_local, PARAMS );
        offspring[block_idx * PARAMS + mut_location] = curand_uniform( &rng_local );
      }

      if( idx == 0 && VERBOSE ) {
        printf( "mutated: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", offspring[block_idx * PARAMS + i] );
        printf( "\n" );
      }

      // ### selection ###
      // == select if better

      // evaluate the offspring
      dtlz1( offspring + block_idx * PARAMS, offspring_fit + block_idx * OBJS, PARAMS, OBJS );

      if( idx == 0 && VERBOSE ) {
        printf( "offspring fit: " );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", offspring_fit[block_idx * OBJS + i] );
        printf( "\n" );
      }

      // compare and copy
      fit_1 = weighted_fitness( objectives + idx * OBJS, x, y );
      fit_2 = weighted_fitness( offspring_fit + block_idx * OBJS, x, y );

      if(fit_2 < fit_1) {
        for (size_t i = 0; i < PARAMS; i++)
          population[i + idx * PARAMS] = offspring[block_idx * PARAMS + i];
        for (size_t i = 0; i < OBJS; i++)
          objectives[i + idx * OBJS] = offspring_fit[block_idx * OBJS + i];
      }

      if( idx == 0 && VERBOSE ) {
        printf( "new ind: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", population[i + idx * PARAMS] );
        printf( "\n" );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", objectives[i + idx * OBJS] );
        printf( "\n" );
      }
    }
    __syncthreads();
  }

  return;
}

/*! \brief main function

  Classic main function. It allocates all memory, generates the population, starts the kernel and collects the results.
  All parameter changes are made via the #define statements
*/
int main() {

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
      //population_h[i * PARAMS + j] = ((float)i)/PARAMS;
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
  write_objectives( objectives_h );
  write_info( elapsedTime );

  // free resources
  free( population_h );
  free( objectives_h );
  ERR( cudaEventDestroy( start ) );
  ERR( cudaEventDestroy( stop ) );

  ERR( cudaFree( population_d ) );
  ERR( cudaFree( objectives_d ) );
}
