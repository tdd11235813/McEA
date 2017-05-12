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

/*! \brief writes the objectives into a file

Writes the given objectives into a file in the JMetal 5.2 file format.
The objectives have to be given one individual after another.
E. g. first all objectives of individual 1, next all of individual 2, ...

\param filename a pointer to the name of the file to write
\param objectives a pointer to the objectives array
\param pop_size the size of the population, that shall be written
\param obj_count the number of objectives per individual
*/
void write_objectives( const char *filename, float *objectives, int pop_size, int obj_count ) {

  FILE *out_file = fopen( filename, "w");
  if(out_file == NULL)
    printf("file: %s cannot be opened.\n", filename);

  for (size_t i = 0; i < pop_size; i++) {
    for (size_t j = 0; j < obj_count; j++) {
      fprintf( out_file, "%f\t", objectives[i*obj_count + j] );
    }
    fprintf( out_file, "\n" );
  }

  fclose( out_file );
}
