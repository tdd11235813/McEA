/*! \file dtlz.cu
  This module contains all used DTLZ functions. Definition of the functions can be found in:
  Deb, Kalyanmoy, et al. "Scalable multi-objective optimization test problems." Evolutionary Computation, 2002. CEC'02. Proceedings of the 2002 Congress on. Vol. 1. IEEE, 2002.
*/

/*!
  Test function. Performs the sum of all params and multiplies it with the respective objectives index.
  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
__device__ void testObjSum( float *params, float *objectives, int param_size, int obj_size ) {

  float param_sum = 0.0;
  for (size_t i = 0; i < param_size; i++) {
    param_sum += params[i];
  }

  for (size_t i = 0; i < obj_size; i++) {
    objectives[i] = param_sum * i;
  }

  return;
}
