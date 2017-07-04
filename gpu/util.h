#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

float randomFloat( void );

void printVector( float *, int );

float get_objective_sum( float *, int , int );

void write_objectives( float *objectives, char *folder );

void write_info( float runtime, char *folder );
#endif
