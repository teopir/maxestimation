#!/bin/bash -x

for i in {1..2}
do
	python3.5 pricing_est.py --nbins=15 --nsamples=150 --suffix=${i}
done


