#!/bin/bash -x

folder=pricing_results_gmm

for i in {1..10}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=100 --suffix=${i} --folder=${folder}
done

for i in {1..10}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=300 --suffix=${i}  --folder=${folder}
done

for i in {1..10}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=200 --suffix=${i} --folder=${folder}
done

for i in {1..10}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=500 --suffix=${i} --folder=${folder}
done

for i in {1..10}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=500 --suffix=${i} --folder=${folder}
done

echo "DONE"
