#include "math.h"
#include <iostream>

using namespace std;

/*! \file dtlz.c
  This module contains all used DTLZ functions. Definition of the functions can be found in:
  Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). Scalable test problems for evolutionary multiobjective optimization (pp. 105-145). Springer London.
*/

/*!
  Test function. Performs the sum of all params and multiplies it with the respective objectives params.
  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
void testObjSum( float *params, float *objectives, int param_size, int obj_size ) {

  float param_sum = 0.0;
  for (int i = 0; i < param_size; i++) {
    param_sum += params[i];
  }

  for (int i = 0; i < obj_size; i++) {
    objectives[i] = param_sum * i;
  }

  return;
}

/*! \brief Function of the DTLZ1 multicriterial optimization problem

  Calculates the objectives for the DTLZ1 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
void dtlz1( float *params, float *objectives, int param_size, int obj_size ) {

		double g = 0.0;
		for (int i = obj_size - 1; i < param_size; i++) {
			g += powf(params[i] - 0.5, 2.0)
					- cosf(20.0 * M_PI * (params[i] - 0.5));
		}
		g = 0.5 * (1.0 + 100.0 * (param_size - obj_size + 1 + g));

    // first iteration is different
    double f = g;
		for (int j = 0; j < obj_size - 1; j++) {
			f *= params[j];
		}
    objectives[0] = f;

    // all others have 1 additional step
		for (int i = 1; i < obj_size; i++) {
			f = g;

			for (int j = 0; j < obj_size - i - 1; j++) {
				f *= params[j];
			}
			f *= 1 - params[obj_size - i - 1];

      objectives[i] = f;
		}

    return;
}

/*! \brief Function of the DTLZ2 multicriterial optimization problem

  Calculates the objectives for the DTLZ2 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
void dtlz2( float *params, float *objectives, int param_size, int obj_size ) {

  double g = 0.0;
  for (int i = obj_size - 1; i < param_size; i++)
    g += powf(params[i] - 0.5,2);

  // different first iteration
  double f = (1 + g);
  for (int j = 0; j < obj_size - 1; j++)
    f *= cosf(params[j] * M_PI / 2);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = (1 + g);
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf(params[j] * M_PI / 2);

    f *= sinf(params[obj_size - i - 1] * M_PI / 2);

    objectives[i] = f;
  }
}

/*! \brief Function of the DTLZ3 multicriterial optimization problem

  Calculates the objectives for the DTLZ3 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
void dtlz3( float *params, float *objectives, int param_size, int obj_size ) {

	double g = 0.0;
	for (int i = obj_size - 1; i < param_size; i++) {
		g += powf(params[i] - 0.5, 2.0)
				- cosf(20.0 * M_PI * (params[i] - 0.5));
	}
	g = 1.0 + 100.0 * (param_size - obj_size + 1 + g);

  // different first iteration
  double f = g;
  for (int j = 0; j < obj_size - 1; j++)
    f *= cosf(params[j] * M_PI / 2);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = g;
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf(params[j] * M_PI / 2);

    f *= sinf(params[obj_size - i - 1] * M_PI / 2);

    objectives[i] = f;
  }
}

/*! \brief Function of the DTLZ4 multicriterial optimization problem

  Calculates the objectives for the DTLZ4 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
void dtlz4( float *params, float *objectives, int param_size, int obj_size ) {

  double g = 0.0;
  double alpha = 100.0;
  for (int i = obj_size - 1; i < param_size; i++)
    g += powf(params[i] - 0.5,2);

  // different first iteration
  double f = (1 + g);
  for (int j = 0; j < obj_size - 1; j++)
    f *= cosf( powf(params[j], alpha) * M_PI / 2);
  objectives[0] = f;

  for (int i = 1; i < obj_size; i++) {
    f = (1 + g);
    for (int j = 0; j < obj_size - i - 1; j++)
      f *= cosf( powf(params[j], alpha) * M_PI / 2);

    f *= sinf( powf(params[obj_size - i - 1], alpha) * M_PI / 2);

    objectives[i] = f;
  }
}

/*! \brief Function of the DTLZ7 multicriterial optimization problem

  Calculates the objectives for the DTLZ7 problem [deb2005scalable], given an array of parameters.

  \param params pointer to array of param values
  \param objectives pointer to objective array
  \param param_size number of elements in the param array
  \param obj_size number of elements in the objective array
*/
void dtlz7( float *params, float *objectives, int param_size, int obj_size ) {

        float g = 0.0;
        float h = 0.0;

        for (int i = obj_size - 1; i < param_size; i++) {
            g += params[i];
        }
        g= 2 + ( 9 * g ) / (param_size - obj_size + 1);


        for (int i = 0; i < obj_size - 1 ; i++)
            objectives[i] = params[i];

        for (int i = 0 ; i < obj_size - 1; i++)
            h += params[i] / g * (1 + sinf(3 * M_PI * params[i]));
        h = obj_size - h;

        objectives[obj_size-1] =  g * h;
}
