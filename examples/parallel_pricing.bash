#!/bin/bash
export OMP_NUM_THREADS=2

N_BINS="2"
N_SAMPLES="100 200 300 400 500"
N_EXPERIMENTS=10

N_PARALLEL="5"

COMMAND="python pricing_est.py --folder=pricing_results_gmm "

EXPERIMENTS=$(seq 1 $N_EXPERIMENTS)
parallel -j $N_PARALLEL $COMMAND ::: $EXPERIMENTS ::: $N_BINS ::: $N_SAMPLES
