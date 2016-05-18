#!/bin/bash -x

function maxproc {
   while [ `jobs | wc -l` -ge 10 ]
   do
      sleep 5
   done
}

for n in 200 300 100
do
	for i in {1..20}
	do
		maxproc; python3.5 pricing_est.py --nbins=15 --nsamples=${n} --suffix=${i} &
	done
done

