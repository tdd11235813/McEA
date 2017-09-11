#ifdef __CUDACC__
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>

#else
#include <random>

#endif


/*! Returns a random float number in [0,1). */
float randomFloat()
{
      return (float)rand()/(float)RAND_MAX;
}
/*! \brief init PRNG

Initializes one pseudo random number generator for each thread.
The same seed is used, but every thread uses a different sequence.

\param[out] state the data structure for the PRNG state
\param[in] seed the seed for initialization

\return nothing
*/
#ifdef __CUDACC__
__global__
void rand_init( curandStatePhilox4_32_10_t  *state, unsigned long seed ) {
  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  curand_init(seed, idx, 0, &state[idx]);
}
#else
/*! \brief init PRNG

Initializes one pseudo random number generator for each thread.

\param[in] seed the seed for initialization

\return nothing
*/
std::default_random_engine rand_init( long seed ) {

  std::default_random_engine generator( seed );
  return generator;
}

#endif

/*! \brief generates a random uniform int

  Takes a value from a uniform distribution in [0, 1] and converts it to an integer in the range [0, values-1].

  \param[in] rand_num the random number to transform
  \param[in] values the number of possible values for the uniform distribution

  \return an integer in the specified range
*/
#ifdef __CUDACC__
__device__
#endif
int trans_uniform_int( float rand_num, int values ) {

    return (int)truncf( rand_num * ( values - 0.000001) );
}

/*! \brief generates a random uniform int

  Draws from a uniform distribution in [0, 1] and converts it to an integer in the range [0, values-1].

  \param[in] state the PRNG state to use
  \param[in] values the number of possible values for the uniform distribution

  \return an integer in the specified range
*/
#ifdef __CUDACC__
__device__
int rnd_uniform_int( curandStatePhilox4_32_10_t  *state, int values ) {

    return trans_uniform_int( curand_uniform( state ), values);
}
#endif
