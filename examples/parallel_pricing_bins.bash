#!/bin/bash
export OMP_NUM_THREADS=2

N_BINS="2 4 6 8 10"
N_SAMPLES="100 200 300 400 500"
N_EXPERIMENTS=50

N_PARALLEL="1"

COMMAND="python pricing_est.py --exclude_weighted=True --folder=pricing_results_gmm_aaai/noweighted "

EXPERIMENTS=$(seq 1 $N_EXPERIMENTS)

parallel -j $N_PARALLEL $COMMAND ::: $EXPERIMENTS ::: $N_BINS ::: $N_SAMPLES
