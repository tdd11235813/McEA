/*! \file mcea.c
  Main algorithm. Starts the calculations via OpenCL.
*/
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>
#include <omp.h>
#include <numeric>
#include <cmath>

// own header files
#include "util.h"
#include "dtlz.h"
#include "config.h"

using namespace std;

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
int get_neighbor(int x, int y, int neighbor_index) {
  // 2D indices
  int n_x = (x + neighbor_index % N_WIDTH - N_RAD + POP_WIDTH + 1) % (POP_WIDTH + 1);
  int n_y = (y + neighbor_index / N_WIDTH - N_RAD + POP_WIDTH) % POP_WIDTH;

  // global index
  return n_x + n_y * (POP_WIDTH + 1);
}

/*! \brief init PRNG

Initializes one pseudo random number generator for each thread.

\param[in] seed the seed for initialization

\return nothing
*/
default_random_engine rand_init( long seed ) {

  default_random_engine generator( seed );
  return generator;
}

/* \brief calculates the weighted fitness

Takes the objective values of the individual at idx and calculates its fitness.
The specific weights for the individual at location x,y in the population are used for weighting.
The weights are applied via the VADS method [hughes2003multiple] to the objective vector.
! This only works for 3 objectives for now !
TODO: implement utopia vector

\param[in] objectives pointer to the objective values of the individual
\param[in] x the x location of the weighting basis (does not have to be the same ind the objectives are from)
\param[in] y the y location of the weighting basis (does not have to be the same ind the objectives are from)

\return the weighted fitness value
*/
double weighted_fitness( float *objectives, int x, int y ) {
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
  double weight_norm[] = { weights[0] / (double)weight_length, weights[1] / (double)weight_length, weights[2] / (double)weight_length };

  // normalize fitness
  float obj_length = sqrt(
      objectives[0] * objectives[0] +
      objectives[1] * objectives[1] +
      objectives[2] * objectives[2] );
  double obj_norm[] = { objectives[0] / (double)obj_length, objectives[1] / (double)obj_length, objectives[2] / (double)obj_length };

  // calculate the fitness
  return obj_length / pow( inner_product( weight_norm, weight_norm + 3, obj_norm, 0.0 ), VADS_SCALE );
  // numerical more stable version
  // takes a little bit more time
  // dont use normalization for that!
  //return exp( (VADS_SCALE + 1) * log( obj_length ) - VADS_SCALE * log( inner_product( weights, weights + 3, objectives, 0.0 ) ) );
}

/*! \brief McEA loop

  This kernel runs the whole algorithm. All data structures have to be set up prior to this.
  It uses the population and performs GENERATIONS generations, consisting of pairing, crossover, mutation, evaluation, and selection on it.
  At the end the population contains the optimized individuals.

  \param[in,out] population an array containing all parameters of the whole population.
  \param[in,out] objectives an array containing all objective values (there will be written some new ones)
  \param[in] rng_state the initialized state of the PRNG to use
*/
void mcea( float *population, float *objectives, default_random_engine rng_state ) {
  float offspring[PARAMS];
  float offspring_fit[OBJS];

  omp_set_num_threads( THREADS );

  // ### evaluation ###
  #pragma omp parallel for
  for (size_t idx = 0; idx < POP_SIZE + 1; idx++) {
      dtlz4( population+idx*PARAMS, objectives+idx*OBJS, PARAMS, OBJS );
  }

  // init random distributions
  uniform_int_distribution<int> uni_params[THREADS];
  for (size_t i = 0; i < THREADS; i++)
    uni_params[i] = uniform_int_distribution<int>( 0, PARAMS );

  uniform_int_distribution<int> uni_neighbors[THREADS];
  for (size_t i = 0; i < THREADS; i++)
    uni_neighbors[i] = uniform_int_distribution<int>( 0, N_WIDTH * N_WIDTH );

  poisson_distribution<int> poisson_mutation[THREADS];
  for (size_t i = 0; i < THREADS; i++)
    poisson_mutation[i] = poisson_distribution<int>( LAMBDA );

  uniform_real_distribution<float> uni_allel[THREADS];
  for (size_t i = 0; i < THREADS; i++)
    uni_allel[i] = uniform_real_distribution<float>(0.0,1.0);

  // main loop
  for (size_t g = 0; g < GENERATIONS; g++) {
    #pragma omp parallel for private( offspring, offspring_fit )
    for (size_t x = 0; x < POP_WIDTH + 1; x++) {
      for (size_t y = 0; y < POP_WIDTH; y++) {

        int idx = x + y * (POP_WIDTH + 1);
        int thread_idx = omp_get_thread_num();

        // ### pairing ###
        // random neighbors
        int neighbor_1 = get_neighbor( x, y, uni_neighbors[thread_idx]( rng_state ) );
        int neighbor_2 = get_neighbor( x, y, uni_neighbors[thread_idx]( rng_state ) );

        // compare neighbors
        double fit_1 = weighted_fitness( objectives + neighbor_1 * OBJS, x, y );
        double fit_2 = weighted_fitness( objectives + neighbor_2 * OBJS, x, y );
        int neighbor_sel = (fit_1 < fit_2)? neighbor_1 : neighbor_2;

        if( idx == 0 && VERBOSE )
          printf("x: %ld, y: %ld, n1: %3d(%.3f), n2: %3d(%.3f), sel: %3d\n", x, y, neighbor_1, fit_1, neighbor_2, fit_2, neighbor_sel);

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
        int x_over_point = uni_params[thread_idx]( rng_state );
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
        int num_mutations = poisson_mutation[thread_idx]( rng_state );
        if( idx == 0 && VERBOSE )
          printf( "mut: %d\n", num_mutations );

        for (size_t i = 0; i < num_mutations; i++) {
          int mut_location = uni_params[thread_idx]( rng_state );
          offspring[mut_location] = uni_allel[thread_idx]( rng_state );
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
        dtlz4( offspring, offspring_fit, PARAMS, OBJS );

        if( idx == 0 && VERBOSE ) {
          printf( "offspring fit: " );
          for (size_t i = 0; i < OBJS; i++)
            printf( "%.2f, ", offspring_fit[i] );
          printf( "\n" );
        }

        // compare and copy
        fit_1 = weighted_fitness( objectives + idx * OBJS, x, y );
        fit_2 = weighted_fitness( offspring_fit, x, y );

        if( idx == 0 && VERBOSE ) {
          printf( "offspring weight: %.5lf\n", fit_2 );
        }

        if(fit_2 < fit_1) {
          for (size_t i = 0; i < PARAMS; i++) {
            #pragma omp atomic write
            population[i + idx * PARAMS] = offspring[i];
          }
          for (size_t i = 0; i < OBJS; i++) {
            #pragma omp atomic write
            objectives[i + idx * OBJS] = offspring_fit[i];
          }
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

  // setup random generator
  unsigned long seed = clock();
  default_random_engine d_state = rand_init( seed );

  // create random population
  srand( time( NULL ) );
  for (size_t i = 0; i < POP_SIZE; i++) {
    for (size_t j = 0; j < PARAMS; j++) {
      population_h[i * PARAMS + j] = randomFloat();
    }
  }

  struct timespec start, finish;
  double elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);

  // start the algorithm
  mcea( population_h, objectives_h, d_state );
  // for (size_t i = 0; i < POP_SIZE; i++) {
  //   dtlz4( population_h + i*PARAMS, objectives_h + i*OBJS, PARAMS, OBJS );
  // }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  printf( "duration: %f\n", elapsed );

  // write the results to file
  write_objectives( objectives_h );
  write_info( elapsed );

  // free resources
  free( population_h );
  free( objectives_h );
}
