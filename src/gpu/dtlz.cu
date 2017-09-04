#include "math_constants.h"
#include "config.h"
#include "error.h"

/*! \file dtlz.cu
  This module contains all used DTLZ functions. Definition of the functions can be found in:
  Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). Scalable test problems for evolutionary multiobjective optimization (pp. 105-145). Springer London.
*/


#if DTLZ_NUM == 0
/*!
  Test function. Performs the sum of all params and multiplies it with the respective objectives params.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

    float param_sum = 0.0;
    for (size_t i = 0; i < param_size; i++) {
      param_sum += params[i * step_size];
    }

    for (size_t i = 0; i < obj_size; i++) {
      objectives[i * step_size] = param_sum * i;
    }

    return;
  }

#elif DTLZ_NUM == 1
/*! \brief Function of the DTLZ1 multicriterial optimization problem

  Calculates the objectives for the DTLZ1 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

		double g = 0.0;
		for (int i = obj_size - 1; i < param_size; i++) {
			g += powf(params[i * step_size] - 0.5, 2.0)
					- cosf(20.0 * CUDART_PI_F * (params[i * step_size] - 0.5));
		}
		g = 0.5 * (1.0 + 100.0 * (param_size - obj_size + 1 + g));

    // first iteration is different
    double f = g;
		for (int j = 0; j < obj_size - 1; j++) {
			f *= params[j * step_size];
		}
    objectives[0] = f;

    // all others have 1 additional step
		for (int i = 1; i < obj_size; i++) {
			f = g;

			for (int j = 0; j < obj_size - i - 1; j++) {
				f *= params[j * step_size];
			}
			f *= 1 - params[(obj_size - i - 1) * step_size];

      objectives[i * step_size] = f;
		}

    return;
}

#elif DTLZ_NUM == 2
/*! \brief Function of the DTLZ2 multicriterial optimization problem

  Calculates the objectives for the DTLZ2 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

  double g = 0.0;
  for (int i = obj_size - 1; i < param_size; i++)
    g += powf(params[i * step_size] - 0.5, 2.0);

  // different first iteration
  double f = (1 + g);
  for (int j = 0; j < obj_size - 1; j++)
    f *= cosf(params[j * step_size] * CUDART_PI_F / 2);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = (1 + g);
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf(params[j * step_size] * CUDART_PI_F / 2);

    f *= sinf(params[(obj_size - i - 1) * step_size] * CUDART_PI_F / 2);

    objectives[i * step_size] = f;
  }
}

#elif DTLZ_NUM == 3
/*! \brief Function of the DTLZ3 multicriterial optimization problem

  Calculates the objectives for the DTLZ3 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

	double g = 0.0;
	for (int i = obj_size - 1; i < param_size; i++) {
		g += powf(params[i * step_size] - 0.5, 2.0)
				- cosf(20.0 * CUDART_PI_F * (params[i * step_size] - 0.5));
	}
	g = 1.0 + 100.0 * (param_size - obj_size + 1 + g);

  // different first iteration
  double f = g;
  for (int j = 0; j < obj_size - 1; j++)
    f *= cosf(params[j * step_size] * CUDART_PI_F / 2);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = g;
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf(params[j * step_size] * CUDART_PI_F / 2);

    f *= sinf(params[(obj_size - i - 1) * step_size] * CUDART_PI_F / 2);

    objectives[i * step_size] = f;
  }
}

#elif DTLZ_NUM == 4
/*! \brief Function of the DTLZ4 multicriterial optimization problem

  Calculates the objectives for the DTLZ4 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

  double g = 0.0;
  double alpha = 100.0;
  for (int i = obj_size - 1; i < param_size; i++)
    g += powf(params[i * step_size] - 0.5,2);

  // different first iteration
  double f = (1 + g);
  for (int j = 0; j < obj_size - 1; j++)
    f *= cos( powf(params[j * step_size], alpha) * CUDART_PI_F / 2);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = (1 + g);
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cos( powf(params[j * step_size], alpha) * CUDART_PI_F / 2);

    f *= sin( powf(params[(obj_size - i - 1) * step_size], alpha) * CUDART_PI_F / 2);

    objectives[i * step_size] = f;
  }
}

#elif DTLZ_NUM == 5
/*! \brief Function of the DTLZ5 multicriterial optimization problem

  Calculates the objectives for the DTLZ5 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

  float g = 0.0;
  float t = 0.0;
  float theta[OBJS-1];

  for (int i = obj_size - 1; i < param_size; i++) {
    g += powf(params[i * step_size]-0.5,2);
  }

  t = CUDART_PI_F /(4 * (1 + g));

  theta[0]= (CUDART_PI_F / 2) * params[0];
  for (int i = 1; i < obj_size - 1 ; i++)
    theta[i]=  t * (1 + 2 * g * params[i * step_size]);

  double f = 1 + g;
  for (int j = 0; j < obj_size - 1; j++)
      f *= cosf(theta[j]);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = (1 + g);
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf(theta[j]);

    f *= sinf(theta[obj_size - i - 1]);

    objectives[i * step_size] = f;
  }
}

#elif DTLZ_NUM == 6
/*! \brief Function of the DTLZ6 multicriterial optimization problem

  Calculates the objectives for the DTLZ6 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size ) {

  float g = 0.0;
  float t = 0.0;
  float theta[OBJS-1];

  for (int i = obj_size - 1; i < param_size; i++)
    g += powf(params[i * step_size],0.1);

  t = CUDART_PI_F /(4 * (1 + g));

  theta[0]= (CUDART_PI_F / 2) * params[0];
  for (int i = 1; i < obj_size - 1 ; i++)
    theta[i]=  t * (1 + 2 * g * params[i * step_size]);

  double f = 1 + g;
  for (int j = 0; j < obj_size - 1; j++)
      f *= cosf(theta[j]);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = (1 + g);
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf(theta[j]);

    f *= sinf(theta[obj_size - i - 1]);

    objectives[i * step_size] = f;
  }
}

#elif DTLZ_NUM == 7
/*! \brief Function of the DTLZ7 multicriterial optimization problem

  Calculates the objectives for the DTLZ7 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
  \param step_size number of elements to skip between 2 parameters (should be the number of threads)
*/
__device__ void dtlz( float *params, float *objectives, int param_size, int obj_size, int step_size) {

        float g = 0.0;
        float h = 0.0;

        for (int i = obj_size - 1; i < param_size; i++) {
            g += params[i * step_size];
        }
        g= 2 + ( 9 * g ) / (param_size - obj_size + 1);


        for (int i = 0; i < obj_size - 1 ; i++)
            objectives[i * step_size] = params[i * step_size];

        for (int i = 0 ; i < obj_size - 1; i++)
            h += params[i * step_size] / g * (1 + sinf(3 * CUDART_PI_F * params[i * step_size]));
        h = obj_size - h;

        objectives[(obj_size-1) * step_size] =  g * h;
}
#endif
