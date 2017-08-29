/*! \file mcea_async.cu
  Main algorithm. Does all the memory management and starts the kernels.
*/
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <string>

// own header files
#include "error.h"
#include "util.h"
#include "dtlz.cuh"
#include "config.h"

using namespace std;

  //! pointers to the dtlz functions
  __device__ void (*dtlz_funcs[])(float*,float*,int,int,int) = { &dtlz1, &dtlz2, &dtlz3, &dtlz4, &dtlz5, &dtlz6, &dtlz7 };

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

/*! \brief generates a random uniform int

  Draws from a uniform distribution in [0, 1] and converts it to an integer in the range [0, values-1].

  \param[in] state the PRNG state to use
  \param[in] values the number of possible values for the uniform distribution

  \return an integer in the specified range
*/
__device__ int rnd_uniform_int( curandStatePhilox4_32_10_t  *state, int values ) {

    return (int)truncf( curand_uniform( state ) * ( values - 0.000001) );
}

/*! \brief calculates the dot product for 2 vectors

   Interprets the values at the pointers x and y as vectors of size 3 and calculates the dot product from them.
   This is only aplicable for vectors of size 3!

   \param[in] x pointer to the first operand
   \param[in] y pointer to the second operand
   \return the scalar value of the dot product
*/
__device__ float inner_product_3( float *x, float *y) {

  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

/*! \brief calculates the weighted fitness

Takes the objective values of the individual at idx and calculates its fitness.
The specific weights for the individual at location x,y in the population are used for weighting.
! This only works for 3 objectives for now !
TODO: for real world problems use the weighted tchebychev method (use utopia vector)

\param[in] objectives pointer to the first objective value of the individual
\param[in] x the x location of the weighting basis (does not have to be the same ind the objectives are from)
\param[in] y the y location of the weighting basis (does not have to be the same ind the objectives are from)
\param[in] offset the distance between two objective values in memory

\return the weighted fitness value
*/
__device__ double weighted_fitness( float *objectives, int x, int y, int offset) {

  // this decides if the individual is on the mirrored side of the population
  // and gives the correction factor for the weights
  int mirror = (x + y < POP_WIDTH)? false : true;

  // calculate weights
  float displacement = (mirror) ? 0.25              : 0.0;
  int _x             = (mirror) ? POP_WIDTH - y - 1 : x;
  int _y             = (mirror) ? POP_WIDTH - x     : y;

  float weights[OBJS];
  weights[0] = (1 - (_x+displacement)/(POP_WIDTH-0.5) - (_y+displacement)/(POP_WIDTH-0.5));
  weights[1] = (_x+displacement)/(POP_WIDTH-0.5);
  weights[2] = (_y+displacement)/(POP_WIDTH-0.5);

  // normalize weight vector
  float weight_length = sqrt(
      weights[0] * weights[0] +
      weights[1] * weights[1] +
      weights[2] * weights[2] );
  float weight_norm[] = { weights[0] / weight_length, weights[1] / weight_length, weights[2] / weight_length };

  // normalize fitness
  float obj_length = sqrt(
      objectives[0]        * objectives[0] +
      objectives[offset]   * objectives[offset] +
      objectives[offset*2] * objectives[offset*2] );

  float obj_norm[] = { 
    objectives[0]        / obj_length,
    objectives[offset]   / obj_length,
    objectives[offset*2] / obj_length };

  // calculate the fitness
  return obj_length / pow( (double)inner_product_3( weight_norm, obj_norm), VADS_SCALE );
  // numerical more stable version
  // takes more time, needs a higher VADS_SCALE
  //return exp( (VADS_SCALE + 1) * log( (double)obj_length ) - VADS_SCALE * log( (double)inner_product_3( weight_norm,  obj_norm) ) );
}

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
  void (*dtlz_ptr)(float*, float*, int, int, int) = dtlz_funcs[DTLZ];
  curandStatePhilox4_32_10_t rng_local;

  // global indices
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);
  // blockwise indices
  int block_idx = (blockDim.x * threadIdx.y + threadIdx.x);

  if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
    rng_local = *(rng_state + idx);
    // ### evaluation ###
    (*dtlz_ptr)( population+idx, objectives+idx, PARAMS, OBJS, POP_SIZE );
  }

  // main loop
  for (size_t g = 0; g < GENERATIONS; g++) {

    if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
      // ### pairing ###
      // random neighbors
      int neighbor_1 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );
      int neighbor_2 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );

      // compare neighbors
      double fit_1 =  weighted_fitness( objectives + neighbor_1, x, y, POP_SIZE );
      double fit_2 =  weighted_fitness( objectives + neighbor_2, x, y, POP_SIZE );
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
      int x_over_point = rnd_uniform_int( &rng_local, PARAMS );
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
      (*dtlz_ptr)( offspring + block_idx , offspring_fit + block_idx, PARAMS, OBJS, BLOCKSIZE );

      if( idx == 0 && VERBOSE ) {
        printf( "offspring fit: " );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", offspring_fit[block_idx + BLOCKSIZE * i] );
        printf( "\n" );
      }

      // compare and copy
      fit_1 =  weighted_fitness( objectives + idx, x, y, POP_SIZE );
      fit_2 =  weighted_fitness( offspring_fit + block_idx, x, y, BLOCKSIZE );

      if( idx == 0 && VERBOSE ) {
        printf( "offspring weight: %.5lf\n", fit_2 );
      }

      if(fit_2 < fit_1) {
        for (size_t i = 0; i < PARAMS; i++)
          population[idx + i * POP_SIZE] = offspring[block_idx + BLOCKSIZE * i];
        for (size_t i = 0; i < OBJS; i++)
          objectives[idx + i * POP_SIZE] = offspring_fit[block_idx + BLOCKSIZE * i];
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
      population_h[i * PARAMS + j] = 1.0; // randomFloat();
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
}
