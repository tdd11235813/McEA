/*! \file mcea.cu
  Main algorithm. Does all the memory management and starts the kernels.
*/
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

// own header files
#include "error.h"
#include "util.h"
#include "dtlz.cuh"

#define POP_WIDTH 10
#define POP_SIZE (POP_WIDTH * (POP_WIDTH + 1))
#define PARAMS 100
#define OBJS 3
#define N_RAD 1
#define N_WIDTH (2 * N_RAD + 1)
#define P_MUT 0.01
#define LAMBDA (P_MUT * PARAMS)

/*! \brief neighbor calculation

  For a given neighbor index this calculates the neighbors global position realtive to the original individual.
  The neighbor index is a number representing the position of the neighbor in the neighborhood of the individual.
  It is organized rowwise starting at the top-left individual in the neighborhood (index = 0). When there is an
  overflow in any of the directions, the global index will be wrapped around to the other side of the population.

  \param x the x position of the original individual
  \param y the y position of the original individual
  \param neighbor_index the position of the neighbor relative to the original. Allowed values depend on the population size.
*/
__device__ int get_neighbor(int x, int y, int neighbor_index) {
  // 2D indices
  int n_x = (x + neighbor_index % N_WIDTH - N_RAD + POP_WIDTH + 1) % (POP_WIDTH + 1);
  int n_y = (y + neighbor_index / N_WIDTH - N_RAD + POP_WIDTH) % POP_WIDTH;

  // global index
  return n_x + n_y * (POP_WIDTH + 1);
}

/*! \brief init PRNG

Initializes the pseudo random number generator.

\param state the data structure for the RNG state
\param seed the seed with which to init the RNG
*/
__global__ void rand_init( curandState *state ) {
  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  unsigned long long seed = (unsigned long long) clock64();
  curand_init(seed, idx, 0, &state[idx]);
}

/* \brief generates a random uniform int

  Draws from a uniform distribution in [0, 1] and converts it to an integer in the range [0, values-1].

  \param state the PRNG state to use
  \param values the number of possible values for the uniform distribution
*/
__device__ int rnd_uniform_int( curandState *state, int values ) {

    return (int)truncf( curand_uniform( state ) * ( values - 0.000001) );
}

/* \brief calculates the weighted fitness

Takes the objective values of the individual at idx and calculates its fitness.
The specific weights for the individual at location x,y in the population are used for weighting.
! This only works for 3 objectives for now !
TODO: for real world problems use the weighted tchebychev method (use utopia vector)

\param objectives pointer to the objective values of the individual
\param x the x location of the weighting basis (does not have to be the same ind the objectives are from)
\param y the y location of the weighting basis (does not have to be the same ind the objectives are from)
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

/*! \brief main kernel

  This kernel runs the whole algorithm. All data structures have to be set up for this.
  TODO: implement algorithm
  \param population an array containing all parameters of the whole population.
  \param objectives an array containing all objective values (there will be written some new ones)
  \param utopia_vec a vector containing the best values for each single objective
  \param rng_state the initialized state of the PRNG to use
*/
__global__ void mcea( float *population, float *objectives, float *utopia_vec, curandState *rng_state ) {
  float offspring[PARAMS];
  float offspring_fit[OBJS];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);


  if( idx < POP_SIZE ) {
    // ### evaluation ###
    dtlz1( population+idx*PARAMS, objectives+idx*OBJS, PARAMS, OBJS );

    // ### pairing ###
    // random neighbors
    int neighbor_1 = get_neighbor( x, y, rnd_uniform_int( rng_state + idx, N_WIDTH * N_WIDTH ) );
    int neighbor_2 = get_neighbor( x, y, rnd_uniform_int( rng_state + idx, N_WIDTH * N_WIDTH ) );

    // compare neighbors
    float fit_1 = weighted_fitness( objectives + neighbor_1 * OBJS, x, y );
    float fit_2 = weighted_fitness( objectives + neighbor_2 * OBJS, x, y );
    int neighbor_sel = (fit_1 < fit_2)? neighbor_1 : neighbor_2;

    if( idx == 99 )
      printf("x: %d, y: %d, n1: %3d(%.3f), n2: %3d(%.3f), sel: %3d\n", x, y, neighbor_1, fit_1, neighbor_2, fit_2, neighbor_sel);

    if( idx == 99 ) {
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
    if( idx == 99 )
      printf( "xover: %d\n", x_over_point );

    for (size_t i = 0; i < PARAMS; i++)
      offspring[i] = (i<x_over_point) ? population[i + idx * PARAMS] : population[i + neighbor_sel * PARAMS];

    if( idx == 99 ) {
      printf( "crossover: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", offspring[i] );
      printf( "\n" );
    }
    // ### mutation ###
    // == uniform mutation
    int num_mutations = curand_poisson( rng_state + idx, LAMBDA );
    if( idx == 99 )
      printf( "mut: %d\n", num_mutations );

    for (size_t i = 0; i < num_mutations; i++) {
      int mut_location = rnd_uniform_int( rng_state + idx, PARAMS );
      offspring[mut_location] = curand_uniform( rng_state + idx );
    }

    if( idx == 99 ) {
      printf( "mutated: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", offspring[i] );
      printf( "\n" );
    }

    // ### selection ###
    // == select if better

    // evaluate the offspring
    dtlz1( offspring, offspring_fit, PARAMS, OBJS );

    if( idx == 99 ) {
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

    if( idx == 99 ) {
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

  curandState *d_state;
  ERR( cudaMalloc( &d_state, POP_SIZE * sizeof(curandState) ) );
  ERR( cudaMalloc( (void**)&population_d, POP_SIZE * PARAMS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&objectives_d, POP_SIZE * OBJS * sizeof(float) ) );
  ERR( cudaMalloc( (void**)&utopia_vec_d, OBJS * sizeof(float) ) );

  // setup random generator
  rand_init<<<1, POP_SIZE>>>( d_state );

  // create random population
  srand( time( NULL ) );
  for (size_t i = 0; i < POP_SIZE; i++) {
    for (size_t j = 0; j < PARAMS; j++) {
      population_h[i][j] = randomFloat();
      //population_h[i][j] = ((float)i)/PARAMS;
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
  dim3 dimBlock(POP_WIDTH + 1, POP_WIDTH);
  mcea<<<1, dimBlock>>>( population_d, objectives_d, utopia_vec_d, d_state );

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
