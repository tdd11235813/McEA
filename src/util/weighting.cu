#ifdef __CUDACC__
#include "../gpu/config.h"
#else
#include "math.h"
#include "../cpu/config.h"
#endif
/*! \brief applies the weights to the given vector

   Interprets the values at the pointers x and y as vectors of size 3 and calculates the dot product from them.
   This is only aplicable for vectors of size 3!

   \param[in] weigths pointer to the weight array
   \param[in] values pointer to the values to multiply with
   \param[in] offset the memory offset for the values of x
   \return the scalar value of the dot product
*/
#ifdef __CUDACC__
__device__
#endif
float weight_multiply( float *x, float *y, int offset ) {

  return x[0] * y[0] + x[offset] * y[1] + x[2*offset] * y[2];
}


/*! \brief calculate weights for individual

Takes the position (x, y) of the individual in the population and calculates the corresponding weights. The weights are stored at the given pointer location

\param[in] x the x location of the weighting basis
\param[in] y the y location of the weighting basis
\param[out] weights a pointer to the location, where the weights will be stored
\param[in] offset the distance between two weight values in memory
*/
#ifdef __CUDACC__
__device__
#endif
void calc_weights( int x, int y, float *weights, int offset) {

  // this decides if the individual is on the mirrored side of the population
  // and gives the correction factor for the weights
  int mirror = (x + y < POP_WIDTH)? false : true;

  // calculate weights
  float displacement = (mirror) ? 0.25              : 0.0;
  int _x             = (mirror) ? POP_WIDTH - y - 1 : x;
  int _y             = (mirror) ? POP_WIDTH - x     : y;

  float tmp_weights[OBJS];
  tmp_weights[0] = (1 - (_x+displacement)/(POP_WIDTH-0.5) - (_y+displacement)/(POP_WIDTH-0.5));
  tmp_weights[1] = (_x+displacement)/(POP_WIDTH-0.5);
  tmp_weights[2] = (_y+displacement)/(POP_WIDTH-0.5);

  // normalize weight vector
  float weight_length = sqrt(
      tmp_weights[0] * tmp_weights[0] +
      tmp_weights[1] * tmp_weights[1] +
      tmp_weights[2] * tmp_weights[2] );

  weights[0]        = tmp_weights[0] / weight_length;
  weights[offset]   = tmp_weights[1] / weight_length;
  weights[offset*2] = tmp_weights[2] / weight_length;

}
/*! \brief calculates the weighted fitness

Takes the objective values of the individual at idx and calculates its fitness.
The specific weights for the individual at location x,y in the population are used for weighting.
! This only works for 3 objectives for now !
TODO: for real world problems use the weighted tchebychev method (use utopia vector)

\param[in] objectives pointer to the first objective value of the individual
\param[in] weights pointer to the first objective value of the individual
\param[in] offset the distance between two objective values in memory

\return the weighted fitness value
*/
#ifdef __CUDACC__
__device__
#endif
double weighted_fitness( float *objectives, float *weights, int offset) {

  // normalize fitness
  float obj_length = sqrt(
      objectives[0]        * objectives[0] +
      objectives[offset]   * objectives[offset] +
      objectives[offset*2] * objectives[offset*2] );

  float obj_norm[] = { 
    objectives[0]        / obj_length,
    objectives[offset]   / obj_length,
    objectives[offset*2] / obj_length };

  // calculate the fitness
#ifdef __CUDACC__
  int weight_offset = BLOCKSIZE;
#else
  int weight_offset = 1;
#endif
  return obj_length / pow( (double)weight_multiply( weights, obj_norm, weight_offset), VADS_SCALE );
}

