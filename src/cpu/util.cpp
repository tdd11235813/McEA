/*! \file util.cpp
  Utilities. Mostly to display results and generate data.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include "config.h"

using namespace std;

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
The param OUTFILE is used as the filename and the extension '.obj' is appended.

\param[in] objectives a pointer to the objectives array
\param[in] folder the folder where the results are saved
\param[in] run the number of the run, that is appended to the files
*/
void write_objectives( float *objectives, string folder, string run) {

  string filename = folder;
  string extension = ".obj";
  filename += OUTFILE;
  filename += string("_");
  filename += run;
  filename += extension;

  FILE *out_file = fopen( filename.c_str(), "w");
  if(out_file == NULL)
    printf("file: %s cannot be opened.\n", filename.c_str() );

  for (size_t i = 0; i < POP_SIZE; i++) {
    for (size_t j = 0; j < OBJS; j++) {
      fprintf( out_file, "%f\t", objectives[i*OBJS + j] );
    }
    fprintf( out_file, "\n" );
  }

  fclose( out_file );
}

/*! \brief writes the runtime and config into a file

Writes a file containing the configuration (the \#define values) and runtime of an optimization run.
The runtime is written in s.
The param OUTFILE is used as the filename and the extension '.obj' is appended.

\param[in] runtime the duration of the calculations (with data copy, without file writing)
\param[in] folder the folder where the results are saved
\param[in] run the number of the run, that is appended to the files
*/
void write_info( float runtime, string folder, string run) {

  string filename = folder;
  string extension = ".info";
  filename += OUTFILE;
  filename += string("_");
  filename += run;
  filename += extension;

  FILE *out_file = fopen( filename.c_str(), "w");
  if(out_file == NULL)
    printf("file: %s cannot be opened.\n", filename.c_str() );

  fprintf( out_file, "name:\t\t%s\n", filename.c_str() );
  fprintf( out_file, "runtime:\t%f ms\n", runtime );
  fprintf( out_file, "dtlz_problem:\t%d\n", DTLZ_NUM );
  fprintf( out_file, "threads:\t%d\n", THREADS );
  fprintf( out_file, "generations:\t%d\n", GENERATIONS );
  fprintf( out_file, "pop_width:\t%d\n", POP_WIDTH );
  fprintf( out_file, "pop_size:\t%d\n", POP_SIZE );
  fprintf( out_file, "param_count\t%d\n", PARAMS );
  fprintf( out_file, "objectives:\t%d\n", OBJS );
  fprintf( out_file, "neighborhood:\t%d\n", N_RAD );
  fprintf( out_file, "mutation_prob:\t%f\n", P_MUT );

  fclose( out_file );
}
