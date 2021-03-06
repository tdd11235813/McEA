'''
==============
3D scatterplot
==============
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import csv


file = sys.argv[1]
points = int(sys.argv[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# read the optimal values
optimal = []
with open('./solutions_dtlz/DTLZ7.3D.pf', 'rb') as csvfile:
    num_lines = sum(1 for line in csvfile)
    csvfile.seek(0)
    fitreader = csv.reader(csvfile, delimiter='\t')
    for row in fitreader:
        if random.random() < float(points) / num_lines:
            optimal.append(map(float, row[:-1]))

optimal = zip(*optimal)

xo = optimal[0]
yo = optimal[1]
zo = optimal[2]
ax.scatter(xo, yo, zo, c=( 1.0, 0.0, 0.0, 0.1 ), marker='.')

# read the fitness values
fitness = []
with open(file, 'rb') as csvfile:
    num_lines = sum(1 for line in csvfile)
    csvfile.seek(0)
    fitreader = csv.reader(csvfile, delimiter='\t')
    for row in fitreader:
        if random.random() < float(points) / num_lines:
            fitness.append(map(float, row))

fitness = zip(*fitness)

xs = fitness[0]
ys = fitness[1]
zs = fitness[2]
ax.scatter(xs, ys, zs, c='b', marker='^')

ax.set_xlabel('crit 1')
ax.set_ylabel('crit 2')
ax.set_zlabel('crit 3')

plt.show()
