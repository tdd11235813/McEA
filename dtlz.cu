
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
