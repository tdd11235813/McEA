/*! \file dtlz.cuh */

#ifndef NEIGHBOR_CUH_INCLUDED
#define NEIGHBOR_CUH_INCLUDED

#ifdef __CUDACC__
__device__ 
#endif
int get_neighbor(int x, int y, int neighbor_index);

#endif
