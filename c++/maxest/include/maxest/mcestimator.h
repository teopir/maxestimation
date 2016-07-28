#ifndef MCESTIMATOR_H
#define MCESTIMATOR_H
#include <iostream>
#include <maxest/niestimator.h>

double compute_product_integral(MaxEstimatorParameters *p);


double g_mc(double *k, size_t dim, void *params);
double g_mc_2(double *k, size_t dim, void *params);

double mc_predict_max(MaxEstApproximator* qf, double minz, double maxz,
                      size_t calls=0, int verbose=0);

double mc_predict_max_2(MaxEstApproximator* qf, double minz, double maxz, int verbose);
#endif // MCESTIMATOR_H



