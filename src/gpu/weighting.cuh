__device__ float weight_multiply( float *x, float *y, int offset );
__device__ void calc_weights( int x, int y, float *weights, const int offset);
__device__ double weighted_fitness( float *objectives, float *weights, int offset);
