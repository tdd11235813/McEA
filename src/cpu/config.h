// ### default values (overwritten by the files in /config)

#ifndef GENERATIONS 
// number of generations that the alg. performs
#define GENERATIONS 10
#endif

#ifndef POP_WIDTH 
// the y-size of the population grid (in individuals)
// can be adjusted at will, scales the memory usage quadratical
#define POP_WIDTH 400
#endif

#ifndef PARAMS 
// the number of parameters for the optimization problen (DTLZ-n)
#define PARAMS 20
#endif

#ifndef N_RAD 
// the radius of the neighborhood around an individual
// the neighborhood is square at all times
#define N_RAD 2
#endif

#ifndef THREADS 
// the number of threads to use in OpenMP
#define THREADS 8
#endif

#ifndef VADS_SCALE 
// the scaling factor for the VADS weighting algorithm
#define VADS_SCALE 100
#endif

#ifndef OUTFILE 
// the base of the filename where the results shall be written
#define OUTFILE out
#endif

#ifndef DTLZ_NUM 
// the number of the DTLZ problem to solve
#define DTLZ_NUM 7
#endif

// ### static values

// the total number of individuals in the population
// the x-size is bigger than the y-size by 1 because of the topology
#define POP_SIZE (POP_WIDTH * (POP_WIDTH + 1))
// the number of optimization goals
// ! don't change this for now (weight calculation is hard coded)
#define OBJS 3
// the width of the neighborhood around an individual
#define N_WIDTH (2 * N_RAD + 1)
// the probability of mutating a gene in 1 generation in 1 individual
#define P_MUT 0.01
// lambda for the poisson distribution of the mutation
#define LAMBDA (P_MUT * PARAMS)
// if true it writes the evolution of individual 0 in the population
// don't use this with a big number of GENERATIONS
#define VERBOSE false
// calculate the index for the DTLZ_NUM
#define DTLZ (DTLZ_NUM - 1)
