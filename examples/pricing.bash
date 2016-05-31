#!/bin/bash -x

export OMP_NUM_THREADS=2
folder=pricing_results_gmm
cmd=nice -n 10 python3.5 pricing_est.py

for i in {1..10}
do
        ${cmd} --nbins=15 --nsamples=100 --suffix=${i} --folder=${folder}
        ${cmd} --nbins=15 --nsamples=200 --suffix=${i} --folder=${folder}
        ${cmd} --nbins=15 --nsamples=300 --suffix=${i} --folder=${folder}
        ${cmd} --nbins=15 --nsamples=400 --suffix=${i} --folder=${folder}
        ${cmd} --nbins=15 --nsamples=500 --suffix=${i} --folder=${folder}
done

echo "DONE"
