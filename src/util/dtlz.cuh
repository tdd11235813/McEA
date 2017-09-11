/*! \file dtlz.cuh */

#ifndef DTLZ_CUH_INCLUDED
#define DTLZ_CUH_INCLUDED

#ifdef __CUDACC__
__device__
#endif
void dtlz( float *, float *, int , int, int);
#ifdef __CUDACC__
__device__
#endif
void dtlz( float *, float *, int, int);

#endif
