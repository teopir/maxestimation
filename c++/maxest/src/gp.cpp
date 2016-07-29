#include "maxest/gp.h"

GP::GP(arma::mat X, arma::mat Y, double l, double sigmaf, double sigman)
    : X(X), Y(Y), l(l), sigmaf(sigmaf), sigman(sigman)
{
    init_kernel(X);
}

GP::GP(char *X_path, char *Y_path, double l, double sigmaf, double sigman)
    : l(l), sigmaf(sigmaf), sigman(sigman)
{
    if (!X.load(X_path, arma::auto_detect))
    {
        std::cout << "Error: unable to read " << X_path << std::endl;
        exit(1);
    }
    if (!Y.load(Y_path, arma::auto_detect))
    {
        std::cout << "Error: unable to read " << Y_path << std::endl;
        exit(1);
    }
    init_kernel(X);

}

double GP::predict(arma::vec& x, double& variance, double& ess)
{
    int npoints = X.n_rows;
    arma::vec k_star(npoints);
    for (unsigned int i = 0; i < npoints; ++i)
    {
        arma::vec row_i = X.row(i);
        k_star(i) = RBFKernel(x, row_i, l, sigmaf);
    }

    arma::vec w = invNoisyKernel * k_star;
    double mean = arma::dot(Y, w);
    variance = RBFKernel(x, x, l, sigmaf) - arma::dot(k_star, w) + sigman * sigman;
    double l1 = 0, l2s = 0;
    for (int i = 0;  i < w.n_elem; ++i) {
        double v = w[i];
        l1 += fabs(v);
        l2s += v*v;
    }
    ess = l1 * l1 / l2s;
    return mean;
}

double GP::predict(double x, double& variance, double& ess)
{
    arma::vec point(1);
    point(0) = x;
    return predict(point, variance, ess);
}

void GP::init_kernel(arma::mat &X)
{
    int npoints = X.n_rows;

    // build kernel matrix
    K.reshape(npoints, npoints);
    for (unsigned int i = 0; i < npoints; ++i)
    {
        arma::vec row_i = X.row(i);
        for (unsigned int j = 0; j < npoints; ++j)
        {
            arma::vec row_j = X.row(j);
            K(i,j) = RBFKernel(row_i, row_j, l, sigmaf);
        }
    }

    // precompute inverse of noisy kernel
    arma::mat T = K + arma::eye(npoints, npoints) * sigman * sigman;
    invNoisyKernel = arma::inv(T);
}


