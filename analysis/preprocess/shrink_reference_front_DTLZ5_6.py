import sys
from sets import Set
import csv

fname_in = sys.argv[1]
fname_out = sys.argv[2]

with open(fname_in, 'r') as file_in, \
    open(fname_out, 'w') as file_out:

    values = Set()

    fitreader = csv.reader(file_in, delimiter=' ')
    for row in fitreader:
        val = tuple(map(float, row[:-1]))
        values.add(val)

    file_out.write('\n'.join('%f\t%f\t%f\t' % x for x in values))
