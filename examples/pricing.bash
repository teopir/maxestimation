#!/bin/bash -x

export OMP_NUM_THREADS=2
folder=pricing_results_gmm
cmd="nice -n 10 python3.5 pricing_est.py"

for i in {1..10}
do
        ${cmd} ${i} 15 100 --folder=${folder}
        ${cmd} ${i} 15 200 --folder=${folder}
        ${cmd} ${i} 15 300 --folder=${folder}
        ${cmd} ${i} 15 400 --folder=${folder}
        ${cmd} ${i} 15 500 --folder=${folder}
done

echo "DONE"
