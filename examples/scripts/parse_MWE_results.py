from __future__ import print_function
import numpy as np
import glob, os
from optparse import OptionParser

op = OptionParser(usage="usage: %prog folder_path",
                  version="%prog 1.0")

(opts, args) = op.parse_args()

if len(args) != 1:
    op.error("wrong number of arguments")

folder = args[0]
OPTIMAL = 2.1717

os.chdir(folder)
print("{:>4}, {:>5}, {:>4}, {:>6}, {:>6}, {:>10}, {:>10}, {:>10}, {:>10}".format("alg", "nsamp", "nrep", "vmean", "vstd", "tmean", "tstd", "mean_verr", "std_verr"))
print("-"*82)
for txtf in glob.glob("MWE*.txt"):
	elements = txtf[0:-4].split("_")
	alg = elements[0]
	nbins = elements[1]
	nsamples = elements[2]
	#print(alg, nbins, nsamples)

	# parse data in the file
	data = np.loadtxt(txtf, delimiter=',')
	means = data.mean(axis=0)
	stds = data.std(axis=0) / np.sqrt(data.shape[0])
	# compute error
	err = np.abs(data[:,1] - OPTIMAL)
	mean_error = err.mean()
	std_error = err.std() / np.sqrt(data.shape[0])

	# print value
	print("{:>4}, {:>5}, {:4d}, {:6.4f}, {:6.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(alg, nsamples, data.shape[0], means[1], stds[1], means[2], stds[2], mean_error, std_error))

print("-"*82)


