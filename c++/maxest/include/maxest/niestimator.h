#ifndef NIESTIMATOR_H
#define NIESTIMATOR_H

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <gsl/gsl_integration.h>
#include <armadillo>

#define ABS_PREC 1e-9
#define REL_PREC 1e-6
#define NBINS 30000


class MaxEstApproximator
{
public:
    virtual double predict(arma::vec& x, double& variance) = 0;
    virtual double predict(double x, double& variance) = 0;
};


struct MaxEstimatorParameters
{
    MaxEstApproximator* qf;
    double mu, sigma;
    double x;
    double space_lower, space_upper;
    int verbose;
};

double pdf(double x, double mu, double sigma);

double cdf(double x, double mu, double sigma);

double product_integral_exponent (double y, void * params);

double prod_int_div_cdf(double x, void * params);

double prob_z_is_max(double z, void* params);

double ni_predict_max(MaxEstApproximator* qf, double minz, double maxz, int verbose);

#endif // NIESTIMATOR_H
