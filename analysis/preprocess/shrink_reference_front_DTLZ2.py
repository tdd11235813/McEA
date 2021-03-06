import math
import sys

fname_in = sys.argv[1]
fname_out = sys.argv[2]
n = 100.0

with open(fname_in, 'r') as file_in, \
    open(fname_out, 'w') as file_out:

    # values = [int(math.ceil(n/2) - 1)]
    values = [9949]

    for line in range(0, int(n)):
        for i in range(int(n) - line):
            index = math.ceil( n / (int(n) - line) * i)
            if( i != 0):
                index -= 1
            values.append(int(n) * line + index)

    lines = file_in.readlines()
    for index in values:
        print('{:d}, '.format(index), end='')
        file_out.write(lines[index])
    print()
