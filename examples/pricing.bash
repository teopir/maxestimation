#!/bin/bash -x

export OMP_NUM_THREADS=2
folder=pricing_results_gmm
cmd="nice -n 10 python3.5 pricing_est.py"

for i in {1..10}
do
        ${cmd} 15 100 ${i} --folder=${folder}
        ${cmd} 15 200 ${i} --folder=${folder}
        ${cmd} 15 300 ${i} --folder=${folder}
        ${cmd} 15 400 ${i} --folder=${folder}
        ${cmd} 15 500 ${i} --folder=${folder}
done

echo "DONE"
