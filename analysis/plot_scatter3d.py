'''
==============
3D scatterplot
==============
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv


file = sys.argv[1]
points = int(sys.argv[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# read the optimal values
optimal = []
with open('/home/est/cloud/promotion/data/DTLZ/DTLZ1.3D.pf', 'rb') as csvfile:
    fitreader = csv.reader(csvfile, delimiter='\t')
    for row in fitreader:
        optimal.append(map(float, row[:-1]))

optimal = zip(*optimal)

xo = optimal[0]
yo = optimal[1]
zo = optimal[2]
ax.scatter(xo, yo, zo, c=( 1.0, 0.0, 0.0, 0.1), marker='.')

# read the fitness values
fitness = []
with open(file, 'rb') as csvfile:
    fitreader = csv.reader(csvfile, delimiter='\t')
    for row in fitreader:
        if len(fitness) < points:
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
