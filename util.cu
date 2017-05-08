/*! \file util.cu
  Utilities. Mostly to display results and generate data.
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*! Returns a random float number in [0,1). */
float randomFloat()
{
      return (float)rand()/(float)RAND_MAX;
}

/*! Prints out a float vector of specifies length to stdout.

  /param[in] vec pointer to the vector to print out
  /param[in] size number of elements to print
*/
void printVector( float *vec, int size) {

  printf( "(" );
  for (size_t i = 0; i < size-1; i++) {
    printf( "%f, ", vec[i] );
  }
  printf( "%f)\n", vec[size-1] );
}

/*! Get the minimal sum of the objectives of an individual.

  \param[in] objectives a pointer to the objective values grouped for each individual
  \param[in] num_population the number of individuals in the population
  \param[in] num_objectives the number of objectives per individual
*/
float get_objective_sum( float *objectives, int num_population, int num_objectives) {
  float min_sum = 50000;
  for (size_t i = 0; i < num_population; i++) {
    float obj_sum = 0;
    for (size_t j = 0; j < num_objectives; j++) {
      obj_sum += objectives[i * num_objectives + j];
    }
    min_sum = (obj_sum < min_sum) ? obj_sum : min_sum;
  }

  return min_sum;
}
