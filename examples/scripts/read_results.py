from __future__ import print_function
import numpy as np
import sys

filename = sys.argv[1]

data = np.loadtxt(filename, delimiter=',')

print(data)

means = data.mean(axis=0)
stds = data.std(axis=0) / np.sqrt(data.shape[0])
print('mean value: {} +- {}'.format(means[1],stds[1]))
print('mean time: {} +- {}'.format(means[2],stds[2]))

