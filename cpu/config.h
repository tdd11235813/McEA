// ### these parameters are defined by the makefile
// number of generations that the alg. performs
// #define GENERATIONS 1000
// the y-size of the population grid (in individuals)
#define POP_WIDTH 400
// the number of parameters for the optimization problen (DTLZ-n)
// can be adjusted at will, scales the memory usage linearly
#define PARAMS 20
// the radius of the neighborhood around an individual
// the neighborhood is square at all times
#define N_RAD 2
// the number of threads to use in OpenMP
#define THREADS 8
// the scaling factor for the VADS weighting algorithm
#define VADS_SCALE 100
// the base of the filename where the results shall be written
#define OUTFILE "test"


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
