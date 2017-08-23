#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

float randomFloat( void );

void printVector( float *, int );

float get_objective_sum( float *, int , int );

void write_objectives( float *objectives, char *folder, char *run );

void write_info( float runtime, char *folde, char *run );

/*! is used for easy access to float4 and int4 variables */
typedef union {
    float4 vec;
    float arr[4];
} float4_union;
typedef union {
    uint4 vec;
    int arr[4];
} int4_union;

#endif
