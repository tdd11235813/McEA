""" takes a list of files filled with fitness values and calculates the overall
PF."""
import sys
import csv
from deap import creator, base, tools


def init_individual(ind_cls):
    return ind_cls()

usage_string = "usage: python bestPF.py <outfile.csv> <file1, [file2, ...]>"

# read the given parameters
outfile = sys.argv[1]
files = sys.argv[2:]

# read the fitness values from the files
fitness = []
for f in files:
    with open(f, 'r') as csvfile:
        fitreader = csv.reader(csvfile, delimiter='\t')
        for row in fitreader:
            fitness.append(map(float, row[:-1]))

# init population
fitness_size = 3
weights = tuple([-1 for _ in range(fitness_size)])
creator.create("FitnessMin", base.Fitness, weights=weights)
creator.create("Individual", object, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual",  # alias
                 init_individual,  # generator function
                 creator.Individual)  # individual class
toolbox.register("population",
                 tools.initRepeat,
                 list,
                 toolbox.individual)
population = toolbox.population(n=len(fitness))

# fill individuals with the fitness values
for ind, fit in zip(population, fitness):
    ind.fitness.values = fit

# pareto_front_ind = tools.sortNondominated(population, len(population), True)
pareto_front_ind = tools.sortLogNondominated(population, len(population), True)

pareto_front = set(map(lambda x: x.fitness.values, pareto_front_ind))

with open(outfile, 'w') as csvfile:
    pfwriter = csv.writer(csvfile, delimiter='\t')
    map(pfwriter.writerow, pareto_front)
