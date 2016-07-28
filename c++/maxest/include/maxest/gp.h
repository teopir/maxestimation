#ifndef GP_H
#define GP_H
#include <armadillo>
#include "maxest/niestimator.h"

inline double RBFKernel(arma::vec& xp, arma::vec& xq, double l, double sigmaf)
{
    double norm = arma::norm(xp-xq, 2);
    double value = sigmaf * sigmaf * exp(- norm * norm / (2*l*l));
    return value;
}

class GP : public MaxEstApproximator
{
public:
    GP(arma::mat X, arma::mat Y, double l, double sigmaf, double sigman);
    GP(char* X_path, char* Y_path, double l, double sigmaf, double sigman);
    double predict(arma::vec& x, double& variance, double &ess);
    double predict(double x, double& variance, double &ess);

public:
    void init_kernel(arma::mat& X);

    arma::mat K, invNoisyKernel;
    arma::mat X, Y;
    double l, sigman, sigmaf;

};

#endif // GP_H
