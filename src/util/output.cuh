#include <string>

using namespace std;

#ifndef UTIL_CUH_INCLUDED
#define UTIL_CUH_INCLUDED

// output functions
void write_objectives( float *objectives, string folder, string run );
void write_info( float runtime, string folder, string run );

#ifdef __CUDACC__
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

#endif
