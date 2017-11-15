/*! \file util.cu
  Utilities to display and write results.
*/
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "../gpu/config.h"

using namespace std;

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
#ifdef __CUDACC__
      out_file.precision(10);
      out_file.setf( ios::fixed, ios::floatfield );
      out_file << objectives[i + j * POP_SIZE] << "\t";
#else
      out_file.precision(10);
      out_file.setf( ios::fixed, ios::floatfield );
      out_file << objectives[i * OBJS + j] << "\t";
#endif
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
