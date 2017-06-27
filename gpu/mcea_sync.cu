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

/* \brief calculates the dot product for 2 vectors

   Interprets the values at the pointers x and y as vectors of size 3 and calculates the dot product from them.
   This is only aplicable for vectors of size 3!

   \param[in] x pointer to the first operand
   \param[in] y pointer to the second operand
   \return the scalar value of the dot product
*/
__device__ float inner_product_3( float *x, float *y) {

  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
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
__device__ double weighted_fitness( float *objectives, int x, int y ) {
  // this decides if the individual is on the mirrored side of the population
  // and gives the correction factor for the weights
  int mirror = (x + y < POP_WIDTH)? false : true;

  // calculate weights
  float offset =  (mirror) ? 0.25 : 0.0;
  int _x  =  (mirror) ? POP_WIDTH - y - 1 : x;
  int _y  =  (mirror) ? POP_WIDTH - x : y;

  float weights[OBJS];
  weights[0] = (1 - (_x+offset)/(POP_WIDTH-0.5) - (_y+offset)/(POP_WIDTH-0.5));
  weights[1] = (_x+offset)/(POP_WIDTH-0.5);
  weights[2] = (_y+offset)/(POP_WIDTH-0.5);

  // normalize weight vector
  float weight_length = sqrt(
      weights[0] * weights[0] +
      weights[1] * weights[1] +
      weights[2] * weights[2] );
  float weight_norm[] = { weights[0] / weight_length, weights[1] / weight_length, weights[2] / weight_length };

  // normalize fitness
  float obj_length = sqrt(
      objectives[0] * objectives[0] +
      objectives[1] * objectives[1] +
      objectives[2] * objectives[2] );
  float obj_norm[] = { objectives[0] / obj_length, objectives[1] / obj_length, objectives[2] / obj_length };

  // calculate the fitness
  return obj_length / pow( (double)inner_product_3( weight_norm, obj_norm), VADS_SCALE );
  // numerical more stable version
  // takes more time, needs a higher VADS_SCALE
  //return exp( (VADS_SCALE + 1) * log( (double)obj_length ) - VADS_SCALE * log( (double)inner_product_3( weight_norm,  obj_norm) ) );
}

/* \brief fitness kernel

   This kernel calculates the initial fitness of all randomly generated individuals.

  \param[in, out] population an array containing all parameters of the whole population.
  \param[in, out] objectives an array containing all objective (= fitness) values 
*/
__global__ void calc_fitness( float *population, float *objectives ) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * (POP_WIDTH + 1);

  if( x < POP_WIDTH + 1 && y < POP_WIDTH )
    dtlz7( population+idx*PARAMS, objectives+idx*OBJS, PARAMS, OBJS );
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
  __shared__ float offspring[PARAMS * BLOCKDIM * BLOCKDIM];
  __shared__ float offspring_fit[OBJS * BLOCKDIM * BLOCKDIM];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int block_idx = (blockDim.x * threadIdx.y + threadIdx.x);
  int idx = x + y * (POP_WIDTH + 1);
  curandStatePhilox4_32_10_t rng_local;

  if( x < POP_WIDTH + 1 && y < POP_WIDTH ) {
    rng_local = *(rng_state + idx);

    // ### pairing ###
    // random neighbors
    int neighbor_1 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );
    int neighbor_2 = get_neighbor( x, y, rnd_uniform_int( &rng_local, N_WIDTH * N_WIDTH ) );

    // compare neighbors
    double fit_1 =  weighted_fitness( objectives_in + neighbor_1 * OBJS, x, y );
    double fit_2 =  weighted_fitness( objectives_in + neighbor_2 * OBJS, x, y );
    int neighbor_sel = (fit_1 < fit_2)? neighbor_1 : neighbor_2;

    if( idx == 0 && VERBOSE )
      printf("x: %d, y: %d, n1: %3d(%.3f), n2: %3d(%.3f), sel: %3d\n", x, y, neighbor_1, fit_1, neighbor_2, fit_2, neighbor_sel);

    if( idx == 0 && VERBOSE ) {
      printf( "original: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", population_in[i + idx * PARAMS] );
      printf( "\n" );
      for (size_t i = 0; i < OBJS; i++)
        printf( "%.2f, ", objectives_in[i + idx * OBJS] );
      printf( "\n" );
    }
    // ### crossover ###
    // == one-point crossover
    int x_over_point = rnd_uniform_int( &rng_local, PARAMS );
    if( idx == 0 && VERBOSE )
      printf( "xover: %d\n", x_over_point );

    for (size_t i = 0; i < PARAMS; i++)
      offspring[block_idx * PARAMS + i] = (i<x_over_point) ? population_in[i + idx * PARAMS] : population_in[i + neighbor_sel * PARAMS];

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
    dtlz7( offspring + block_idx * PARAMS, offspring_fit + block_idx * OBJS, PARAMS, OBJS );

    if( idx == 0 && VERBOSE ) {
      printf( "offspring fit: " );
      for (size_t i = 0; i < OBJS; i++)
        printf( "%.2f, ", offspring_fit[block_idx * OBJS + i] );
      printf( "\n" );
    }

    // compare and copy
    fit_1 =  weighted_fitness( objectives_in + idx * OBJS, x, y );
    fit_2 =  weighted_fitness( offspring_fit + block_idx * OBJS, x, y );

    if( idx == 0 && VERBOSE )
      printf( "offspring weight: %.5lf\n", fit_2 );

    if(fit_2 < fit_1) {
      for (size_t i = 0; i < PARAMS; i++)
        population_out[i + idx * PARAMS] = offspring[block_idx * PARAMS + i];
      for (size_t i = 0; i < OBJS; i++)
        objectives_out[i + idx * OBJS] = offspring_fit[block_idx * OBJS + i];
    }
    else {
      for (size_t i = 0; i < PARAMS; i++)
        population_out[i + idx * PARAMS] = population_in[i + idx * PARAMS];
      for (size_t i = 0; i < OBJS; i++)
        objectives_out[i + idx * OBJS] = objectives_in[i + idx * OBJS];
    }


    if( idx == 0 && VERBOSE ) {
      printf( "new ind: " );
      for (size_t i = 0; i < PARAMS; i++)
        printf( "%.2f, ", population_out[i + idx * PARAMS] );
      printf( "\n" );
      for (size_t i = 0; i < OBJS; i++)
        printf( "%.2f, ", objectives_out[i + idx * OBJS] );
      printf( "\n" );
    }

    *(rng_state + idx) = rng_local;
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
  write_objectives( objectives_h );
  write_info( elapsedTime );

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
