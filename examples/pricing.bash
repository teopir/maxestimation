#!/bin/bash -x

mkdir -p pricing_results/logs
for i in {1..20}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=300 --suffix=${i}
#> pricing_results/logs/15_300_${i}.log
done

for i in {1..20}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=200 --suffix=${i}
#> pricing_results/logs/15_200_${i}.log
done

for i in {1..20}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=500 --suffix=${i}
#> pricing_results/logs/15_100_${i}.log
done

for i in {1..20}
do
        python3.5 pricing_est.py --nbins=15 --nsamples=100 --suffix=${i}
#> pricing_results/logs/15_100_${i}.log
done


echo "DONE"
