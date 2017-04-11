#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float randomFloat()
{
      return (float)rand()/(float)RAND_MAX;
}

void printVector( float *vec, int size) {

  printf( "(" );
  for (size_t i = 0; i < size-1; i++) {
    printf( "%f, ", vec[i] );
  }
  printf( "%f)\n", vec[size-1] );
}
