#!/bin/bash -x

for i in {900..999}
do
	for b in {5..50..5}
	do
		for d in 100 200 300 400 500
		do
			time python3.5 pricing_est.py --nbins=${b} --nsamples=${d} --suffix=${i} --exclude_weighted=True --folder='pricing_results/noweighted'
		done
	done
done
