/*! \file mcea.cpp
  Main algorithm. Starts the calculations via OpenMP.
*/
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>
#include <iostream>
#include <omp.h>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <signal.h>

// own header files
#include "../util/output.h"
#include "../util/dtlz.h"
#include "../util/neighbor.h"
#include "../util/weighting.h"
#include "../util/random.h"
#include "config.h"

using namespace std;

// global flag, used to stop calculations
sig_atomic_t timer_expired = 0;

/*! \brief handler for the ALRM signal
 *
 * This manipulates the timer_expired flag, so that all threads stop at the next appropriate time.
 * Especially offsprings that are created after the call to this handler are not processed.
 * This means they can not become part of the population.
 *
 * \param[in] sig not used
 */
void sigalrm_handler(int sig)
{
  timer_expired = 1;
}

/*! \brief McEA loop

  This kernel runs the whole algorithm. All data structures have to be set up prior to this.
  It uses the population and performs generations until the stopping criterion is reached, consisting of pairing, crossover, mutation, evaluation, and selection on it.
  The available stopping criteria are GENERATIONS and TIME, for more information on that, read config.h.
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
      dtlz( population+idx*PARAMS, objectives+idx*OBJS, PARAMS, OBJS, 1 );
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
#if STOPTYPE == GENERATIONS
  // stop after number of generations
  for (size_t g = 0; g < STOPVALUE; g++) {
    if ( g%10 == 0 )
#elif STOPTYPE == TIME
  // set the handler to stop the run
  signal(SIGALRM, &sigalrm_handler);
  alarm(STOPVALUE);

  // stop after timer expired
  while(true) {
    if(timer_expired)
      break;
#else
    cout << "no valid STOPTYPE. doing just one generation." << endl;
    {
#endif

    #pragma omp parallel for private( offspring, offspring_fit ) shared( timer_expired )
    for (size_t x = 0; x < POP_WIDTH + 1; x++) {
      for (size_t y = 0; y < POP_WIDTH; y++) {

        int idx = x + y * (POP_WIDTH + 1);
        int thread_idx = omp_get_thread_num();
        float weights[OBJS];
        calc_weights(x, y, weights, 1);

        // ### pairing ###
        // random neighbors
        int neighbor_1 = get_neighbor( x, y, uni_neighbors[thread_idx]( rng_state ) ); int neighbor_2 = get_neighbor( x, y, uni_neighbors[thread_idx]( rng_state ) );

        // compare neighbors
        double fit_1 = weighted_fitness( objectives + neighbor_1 * OBJS, weights, 1 );
        double fit_2 = weighted_fitness( objectives + neighbor_2 * OBJS, weights, 1 );
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

        for (int i = 0; i < PARAMS; i++)
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

        for (int i = 0; i < num_mutations; i++) {
          int mut_location = uni_params[thread_idx]( rng_state );
          offspring[mut_location] = uni_allel[thread_idx]( rng_state );
        }

        if( idx == 0 && VERBOSE ) {
          printf( "mutated: " );
          for (size_t i = 0; i < PARAMS; i++)
            printf( "%.2f, ", offspring[i] );
          printf( "\n" );
        }

        if(!timer_expired) {
          // ### selection ###
          // == select if better

          // evaluate the offspring
          dtlz( offspring, offspring_fit, PARAMS, OBJS, 1 );

          if(idx == 0 && VERBOSE) {
            printf( "offspring fit: " );
            for (size_t i = 0; i < OBJS; i++)
              printf( "%.2f, ", offspring_fit[i] );
            printf( "\n" );
          }

          // compare and copy
          fit_1 = weighted_fitness( objectives + idx * OBJS, weights, 1 );
          fit_2 = weighted_fitness( offspring_fit, weights, 1 );

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
  }

  return;
}

/*! \brief main function

  Classic main function. It allocates all memory, generates the population, starts the kernel and collects the results.
  All parameter changes are made via the \#define statements
*/
int main( int argc, char *argv[] ) {

  // get the output folder
  string folder = string("");
  string run = string("0");
  if(argc > 1) {
    folder = string(argv[1]);
    run = string(argv[2]);  
  }

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
  //   (*dtlz_ptr[DTLZ])( population_h + i*PARAMS, objectives_h + i*OBJS, PARAMS, OBJS );
  // }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec) * 1000;
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000.0;
  printf( "duration: %lf\n", elapsed );

  // write the results to file
  write_objectives( objectives_h, folder, run);
  write_info( elapsed, folder, run );

  // free resources
  free( population_h );
  free( objectives_h );
}
