// type definition for the different stoptypes of McEA
// (! don't define 0, because it is also the default value of undefined parameters)
// - GENERATIONS: number of generations that the alg. performs
#define GENERATIONS 1
// - TIME: the time in seconds, after which the algorithm shall be aborted (!only in CPU variant for now)
#define TIME 2

// ### default values (overwritten by the files in /config)
#ifndef STOPTYPE 
// Criterium that stops the generation loop. The available ones are documented above.
#define STOPTYPE GENERATIONS
#endif

#ifndef STOPVALUE 
// the value that is used to determine, when McEA stops (depends on the STOPTYPE)
#define STOPVALUE 10
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
#define OUTFILE "out"
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
// the size of one block in a dimension
#define BLOCKDIM 16
// the size of one whole block
#define BLOCKSIZE (BLOCKDIM * BLOCKDIM)
