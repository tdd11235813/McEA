/*! \file util.cu
  Utilities. Mostly to display results and generate data.
*/
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
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
\param[in] run the number and type of the run, that is appended to the files
*/
void write_objectives( float *objectives, string folder, string run ) {

  ostringstream filename;
  filename << folder << OUTFILE << "_" << run << ".obj";

  ofstream out_file;
  out_file.open(filename.str().c_str());
  if(!out_file.is_open()) {
    cout << "file: " << filename.str() << " cannot be opened.\n";
    return;
  }

  for (size_t i = 0; i < POP_SIZE; i++) {
    for (size_t j = 0; j < OBJS; j++) {
      out_file << objectives[i + j * POP_SIZE] << "\t";
    }
    out_file << "\n";
  }

  out_file.close();
}

/*! \brief writes the runtime and config into a file

Writes a file containing the configuration (the \#define values) and runtime of an optimization run.
The runtime is written in ms.
The param OUTFILE is used as the filename and the extension '.obj' is appended.

\param[in] runtime the duration of the calculations (with data copy, without file writing)
\param[in] folder the folder where the results are saved
\param[in] run the number and type of the run, that is appended to the files
*/
void write_info( float runtime, string folder, string run ) {

  ostringstream filename;
  filename << folder << OUTFILE << "_" << run << ".info";

  ofstream out_file;
  out_file.open (filename.str().c_str());
  if(!out_file.is_open()) {
    cout << "file: " << filename.str() << " cannot be opened.\n";
    return;
    }

  out_file << "name:\t\t" << filename.str() << "\n";
  out_file << "runtime:\t" << fixed << setprecision(2) << runtime << " ms\n";
  out_file << "dtlz_problem:\t" << DTLZ_NUM << "\n";
  out_file << "generations:\t" << GENERATIONS << "\n";
  out_file << "pop_width:\t" << POP_WIDTH << "\n";
  out_file << "pop_size:\t" << POP_SIZE << "\n";
  out_file << "param_count\t" << PARAMS << "\n";
  out_file << "objectives:\t" << OBJS << "\n";
  out_file << "neighborhood:\t" << N_RAD << "\n";
  out_file << "mutation_prob:\t" << P_MUT << "\n";

  out_file.close();
}
