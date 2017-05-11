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

// number of generations that the alg. performs
#define GENERATIONS 100
// the y-size of the population grid (in individuals)
#define POP_WIDTH 127
// the total number of individuals in the population
// the x-size is bigger than the y-size by 1 because of the topology
#define POP_SIZE (POP_WIDTH * (POP_WIDTH + 1))
// the number of parameters for the optimization problen (DTLZ-n)
// can be adjusted at will, scales the memory usage linearly
#define PARAMS 20
// the number of optimization goals
// ! don't change this for now (weight calculation is hard coded)
#define OBJS 3
// the radius of the neighborhood around an individual
// the neighborhood is square at all times
#define N_RAD 2
// the width of the neighborhood around an individual
#define N_WIDTH (2 * N_RAD + 1)
// the probability of mutating a gene in 1 generation in 1 individual
#define P_MUT 0.01
// lambda for the poisson distribution of the mutation
#define LAMBDA (P_MUT * PARAMS)
// if true it writes the evolution of individual 0 in the population
// don't use this with a big number of GENERATIONS
#define VERBOSE false

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
  float offspring[PARAMS];
  float offspring_fit[OBJS];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);


  if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
    // ### evaluation ###
    dtlz1( population+idx*PARAMS, objectives+idx*OBJS, PARAMS, OBJS );

    // main loop
    for (size_t g = 0; g < GENERATIONS; g++) {

      // ### pairing ###
      // random neighbors
      int neighbor_1 = get_neighbor( x, y, rnd_uniform_int( rng_state + idx, N_WIDTH * N_WIDTH ) );
      int neighbor_2 = get_neighbor( x, y, rnd_uniform_int( rng_state + idx, N_WIDTH * N_WIDTH ) );

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
      int x_over_point = rnd_uniform_int( rng_state + idx, PARAMS );
      if( idx == 0 && VERBOSE )
        printf( "xover: %d\n", x_over_point );

      for (size_t i = 0; i < PARAMS; i++)
        offspring[i] = (i<x_over_point) ? population[i + idx * PARAMS] : population[i + neighbor_sel * PARAMS];

      if( idx == 0 && VERBOSE ) {
        printf( "crossover: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", offspring[i] );
        printf( "\n" );
      }
      // ### mutation ###
      // == uniform mutation
      int num_mutations = curand_poisson( rng_state + idx, LAMBDA );
      if( idx == 0 && VERBOSE )
        printf( "mut: %d\n", num_mutations );

      for (size_t i = 0; i < num_mutations; i++) {
        int mut_location = rnd_uniform_int( rng_state + idx, PARAMS );
        offspring[mut_location] = curand_uniform( rng_state + idx );
      }

      if( idx == 0 && VERBOSE ) {
        printf( "mutated: " );
        for (size_t i = 0; i < PARAMS; i++)
          printf( "%.2f, ", offspring[i] );
        printf( "\n" );
      }

      // ### selection ###
      // == select if better

      // evaluate the offspring
      dtlz1( offspring, offspring_fit, PARAMS, OBJS );

      if( idx == 0 && VERBOSE ) {
        printf( "offspring fit: " );
        for (size_t i = 0; i < OBJS; i++)
          printf( "%.2f, ", offspring_fit[i] );
        printf( "\n" );
      }

      // compare and copy
      fit_1 = weighted_fitness( objectives + idx * OBJS, x, y );
      fit_2 = weighted_fitness( offspring_fit, x, y );

      if(fit_2 < fit_1) {
        for (size_t i = 0; i < PARAMS; i++)
          population[i + idx * PARAMS] = offspring[i];
        for (size_t i = 0; i < OBJS; i++)
          objectives[i + idx * OBJS] = offspring_fit[i];
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

      __syncthreads();
    }
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
  dim3 dimBlock(32, 32);
  dim3 dimGrid(ceil((POP_WIDTH + 1) / 32.0) , ceil(POP_WIDTH / 32.0));
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

  // search the minima
  float min_sum = get_objective_sum( objectives_h, POP_SIZE, OBJS );
  printf("min sum: %.2f\n", min_sum);

  // free resources
  free( population_h );
  free( objectives_h );
  free( d_state );
  ERR( cudaEventDestroy( start ) );
  ERR( cudaEventDestroy( stop ) );

  ERR( cudaFree( population_d ) );
  ERR( cudaFree( objectives_d ) );
}
