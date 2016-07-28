#include <iostream>
#include <chrono>
#include <maxest/niestimator.h>
#include <maxest/mcestimator.h>

using namespace std;


class QF : public MaxEstApproximator
{
public:

    double predict(double x, double& var, double& ess)
    {
        var = 0.001;
        ess = 1.0;
        return x*(1-x);
    }

    double predict(arma::vec& x, double& variance, double& ess)
    {
        return predict(x[0], variance, ess);
    }

};

int
main(int argc, char *argv[])
{
    QF prova;
    int verbose = 0;

    auto start = chrono::steady_clock::now();
    double val = ni_predict_max(&prova, -1., 1., verbose);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "max = " << val << std::endl;
    std::cout << chrono::duration <double, milli> (diff).count() << " ms" << std::endl;


    start = chrono::steady_clock::now();
    val = mc_predict_max(&prova, -1., 1., 0, verbose);
    end = chrono::steady_clock::now();
    diff = end - start;

    std::cout << "max = " << val << std::endl;
    std::cout << chrono::duration <double, milli> (diff).count() << " ms" << std::endl;
    return 0;
}

