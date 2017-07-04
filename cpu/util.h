#include <string>

using namespace std;

#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

float randomFloat( void );

void printVector( float *, int );

float get_objective_sum( float *, int , int );

void write_objectives( float *objectives, string folder );

void write_info( float runtime, string folder );
#endif
