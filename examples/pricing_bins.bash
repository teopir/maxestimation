#!/bin/bash -x

folder='pricing_results_gmm/noweighted_gmm'

for i in {900..910}
do
	for b in {2..10..2}
	do
		for d in 100 200 300 400 500
		do
			time python3.5 pricing_est.py --nbins=${b} --nsamples=${d} --suffix=${i} --exclude_weighted=True --folder=${folder}
		done
	done
done
