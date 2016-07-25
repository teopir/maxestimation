#include <iostream>
#include <chrono>
#include <maxest/estimator.h>

using namespace std;


class QF : public MaxEstApproximator {
public:

    double predict(double x, double& var) {
        var = 0.001;
        return x*(1-x);
    }

    double predict(arma::vec& x, double& variance) {
        return predict(x[0], variance);
    }

};

int
main(int argc, char *argv[])
{
    QF prova;
    int verbose = 0;
    auto start = chrono::steady_clock::now();
    double val = predict_max(&prova, -1., 1., verbose);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;

    std::cout << "max = " << val << std::endl;
    std::cout << chrono::duration <double, milli> (diff).count() << " ms" << std::endl;
    return 0;
}

