#ifndef WEIGHTING_CUH_INCLUDED
#define WEIGHTING_CUH_INCLUDED

#ifdef __CUDACC__
__device__
#endif
float weight_multiply( float *x, float *y, int offset );
#ifdef __CUDACC__
__device__
#endif
void calc_weights( int x, int y, float *weights, const int offset);
#ifdef __CUDACC__
__device__
#endif
float weighted_fitness( float *objectives, float *weights, int offset);

#endif
