#ifndef __CUDACC__
#include <random>
#endif

#ifndef RANDOM_CUH_INCLUDED
#define RANDOM_CUH_INCLUDED

// non CUDA
float randomFloat( void );
#ifdef __CUDACC__
__device__
#endif
int trans_uniform_int( float rand_num, int values );

// CUDA only
#ifdef __CUDACC__
__global__
void rand_init( curandStatePhilox4_32_10_t  *state, unsigned long seed );
__device__
int rnd_uniform_int( curandStatePhilox4_32_10_t  *state, int values );
#else
std::default_random_engine rand_init( long seed );
#endif

#endif
