#include <iostream>
#include <chrono>
#include <armadillo>
#include <maxest/niestimator.h>
#include <maxest/mcestimator.h>
#include <maxest/gp.h>

using namespace std;

void help()
{
    cout << "./maxestfromgp " << endl;
    cout << "Inputs: X, y, l, sigmaf, sigman, zmin, zmax" << endl;
}

int
main(int argc, char *argv[])
{
    std::cout << "---------------" << std::endl;
    /*
     * inputs: X, y, l, sigmaf, sigman, zmin, zmax
     */
    if (argc < 8)
    {
        cout << "Error: not enough input arguments" << endl << endl;
        help();
        exit(1);
    }

    char* outfile = "res.dat";
    if (argc == 9)
    {
        outfile = argv[8];
    }
    std::cout << "output will be written in " << outfile << std::endl;

    double l = atof(argv[3]);
    double sigmaf = atof(argv[4]);
    double sigman = atof(argv[5]);

    double minz = atof(argv[6]);
    double maxz = atof(argv[7]);

    GP gp(argv[1], argv[2], l, sigmaf, sigman);
    std::cout << "created GP" << std::endl;

    //    double var, ess;
    //    double pred = gp.predict(.5, var, ess);
    //    cout << "pred: " << pred << ", var: " << var << ", std: " << sqrt(var) << ", ess: " << ess << endl;

    int verbose = 0;
    auto start = chrono::steady_clock::now();
    double val = ni_predict_max(&gp, minz, maxz, verbose);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "max = " << val << " in ";
    std::cout << chrono::duration <double, milli> (diff).count() << " ms" << std::endl;
    ofstream f(outfile);
    if (f.is_open())
    {
        f << val << std::endl;
    }

//    start = chrono::steady_clock::now();
//    val = mc_predict_max(&gp, minz, maxz, 0, verbose);
//    end = chrono::steady_clock::now();
//    diff = end - start;

//    std::cout << "max = " << val << " in ";
//    std::cout << chrono::duration <double, milli> (diff).count() << " ms" << std::endl;
    return 0;
}

