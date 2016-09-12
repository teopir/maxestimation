from __future__ import print_function
import numpy as np
import glob, os
from optparse import OptionParser
import pandas as pd

op = OptionParser(usage="usage: %prog folder_path",
                  version="%prog 1.0")

op.add_option("--real_max", default=2.1717,
              dest="OPTIMAL", type="float",
              help="Real expected maximum value.")

op.add_option("--abs_err", default=False,
              dest="abs_err",
              help="Compute mean value of absolute error.")

(opts, args) = op.parse_args()

if len(args) != 1:
    op.error("wrong number of arguments")

folder = args[0]
# OPTIMAL = 2.1717
# OPTIMAL = 1.824
OPTIMAL = opts.OPTIMAL
abs_err = opts.abs_err

os.chdir(folder)
print("{:>4}, {:>5}, {:>4}, {:>6}, {:>6}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format("alg", "nsamp", "nrep", "vmean", "vstd", "tmean", "tstd", "mean_verr", "std_verr", "var_err"))
print("-"*94)
results = list()
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
	err = data[:,1] - OPTIMAL
	if abs_err == True:
		err = np.abs(err)
	#err = err[np.where(err < 1)[0]]
	mean_error = err.mean()
	std_error_mean = err.std() / np.sqrt(data.shape[0])
	# compute variance
	var_err = err.var()

	# print value
	print("{:>4}, {:>5}, {:4d}, {:6.4f}, {:6.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(alg, nsamples, data.shape[0], means[1], stds[1], means[2], stds[2], mean_error, std_error_mean, var_err))
	results.append([alg, nsamples, data.shape[0], means[1], stds[1], means[2], stds[2], mean_error, std_error_mean, var_err])

print("-"*94)
df = pd.DataFrame(results, columns=['Algorithm', 'Nbins', 'NSamples', 'NReps', 'MaxMean', 'MaxStd', 'TimeMean', 'TimeStd', 'ErrMaxMean', 'ErrMaxStd', 'VarErr'])


