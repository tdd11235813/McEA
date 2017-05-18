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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fitness = []
with open(file, 'rb') as csvfile:
    fitreader = csv.reader(csvfile, delimiter='\t')
    for row in fitreader:
        fitness.append(map(float, row[:-1]))

fitness = zip(*fitness)


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = fitness[0]
ys = fitness[1]
zs = fitness[2]
ax.scatter(xs, ys, zs, c='b', marker='^')

ax.set_xlabel('crit 1')
ax.set_ylabel('crit 2')
ax.set_zlabel('crit 3')

plt.show()
