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
  /param vec pointer to the vector to print out
  /param size number of elements to print
*/
void printVector( float *vec, int size) {

  printf( "(" );
  for (size_t i = 0; i < size-1; i++) {
    printf( "%f, ", vec[i] );
  }
  printf( "%f)\n", vec[size-1] );
}
