#include <maxest/mcestimator.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include <chrono>

double compute_product_integral(MaxEstimatorParameters *p)
{

    gsl_function F;
    F.function = &product_integral_exponent;
    F.params = p;
    double result, error;
    size_t intervals;

    auto start = std::chrono::steady_clock::now();
    // Compute product integral
    gsl_integration_cquad_workspace* w = gsl_integration_cquad_workspace_alloc (NBINS);

    gsl_integration_cquad (&F, p->space_lower, p->space_upper, ABS_PREC, REL_PREC,
                           w, &result, &error, &intervals);
    double pi = exp(result);

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    if (p->verbose > 1)
    {
        std::cout << "Finished product integral computation..." << std::endl;
        printf ("pi              = % .18f\n", pi);
        printf ("estimated error = % .18f\n", error);
        printf ("intervals       = %zu\n", intervals);
        std::cout << "duration        = " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    }

    gsl_integration_cquad_workspace_free (w);

    return pi;
}


double g_mc(double *k, size_t dim, void *params)
{
    auto start = std::chrono::steady_clock::now();

    double z = k[0];

    MaxEstimatorParameters *p = (MaxEstimatorParameters*) params;
    MaxEstApproximator* gp = p->qf;

    double var_z, mu_z, sigma_z, ns;
    mu_z = gp->predict(z, var_z);
    ns = gp->get_noise_sigma();
    sigma_z = sqrt(var_z - ns*ns);

    double w_z = 0.0;

    int i, nbPoints = 8*sigma_z*150;
    // std::cout << z << ", " << nbPoints << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> n01(0.0,1.0);

    for (i = 0; i < nbPoints; ++i)
    {
        double x, cdf_x, pi;
        x = sigma_z * n01(generator) + mu_z;
        // per usare il parallelo allocate un nuovo MaxEstimatorParameters
        p->x = x; //set current point

        cdf_x = cdf(x, mu_z, sigma_z);
        pi = compute_product_integral(p);

        w_z += pi / cdf_x;
    }
    w_z /= nbPoints;

    double value = mu_z * w_z;

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    //std::cout << "duration        = " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

    return value;
}

void
display_results (char *title, double result, double error)
{
    std::cout << title << " ==================\n";
    std::cout << "result = " << result << std::endl;
    std::cout << "sigma  = " << error << std::endl;
}

double mc_predict_max(MaxEstApproximator *qf, double minz, double maxz,
                      size_t calls, int verbose)
{
    MaxEstimatorParameters pl;
    pl.qf = qf;
    pl.space_lower = minz;
    pl.space_upper = maxz;
    pl.verbose = verbose;

    double res, err;

    double xl[1] = { minz};
    double xu[1] = { maxz};

    const gsl_rng_type *T;
    gsl_rng *r;

    gsl_monte_function G = { &g_mc, 1, &pl };

    if (calls == 0)
    {
        calls = (maxz-minz) * 150;
    }

    gsl_rng_env_setup ();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    {
        gsl_monte_miser_state *s = gsl_monte_miser_alloc (1);
        gsl_monte_miser_integrate (&G, xl, xu, 1, calls, r, s,
                                   &res, &err);
        gsl_monte_miser_free (s);

        display_results ("miser", res, err);
    }

    //    {
    //    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (1);

    //    gsl_monte_vegas_integrate (&G, xl, xu, 1, 1000, r, s,
    //                               &res, &err);
    //    display_results ("vegas warm-up", res, err);

    //    printf ("converging...\n");

    //    do
    //    {
    //        gsl_monte_vegas_integrate (&G, xl, xu, 1, calls/5, r, s,
    //                                   &res, &err);
    //        std::cout << "result = " << res << " sigma = " << err <<
    //                     " chisq/dof = " << gsl_monte_vegas_chisq (s) << std::endl;
    //    }
    //    while (fabs (gsl_monte_vegas_chisq (s) - 1.0) > 0.5);

    //    display_results ("vegas final", res, err);
    //    gsl_monte_vegas_free (s);
    //    }

    gsl_rng_free (r);

    return res;
}


double g_mc_2(double *k, size_t dim, void *params)
{
    double x = k[0];

    MaxEstimatorParameters *p = (MaxEstimatorParameters*) params;
    MaxEstApproximator* gp = p->qf;
    p->x = x;

    double w_z = 0.0;
    double pi = compute_product_integral(p);

    int i, nbPoints = 1000;
    // std::cout << z << ", " << nbPoints << std::endl;
    arma::vec points = arma::linspace(p->space_lower, p->space_upper, nbPoints);
    for (i = 0; i < nbPoints; ++i)
    {

        double z = points[i];

        double var_z, mu_z, sigma_z, ns;
        mu_z = gp->predict(z, var_z);
        ns = gp->get_noise_sigma();
        sigma_z = sqrt(var_z - ns*ns);

        double pdf_z, cdf_z;
        pdf_z = pdf(x, mu_z, sigma_z);
        cdf_z = cdf(x, mu_z, sigma_z);

        w_z += mu_z * pdf_z / cdf_z;
    }
    w_z /= nbPoints;

    double value = pi * w_z;
    return value;
}


double mc_predict_max_2(MaxEstApproximator *qf, double minz, double maxz, int verbose)
{
    MaxEstimatorParameters pl;
    pl.qf = qf;
    pl.space_lower = minz;
    pl.space_upper = maxz;
    pl.verbose = verbose;

    double res, err;

    double xl[1] = {-10000};
    double xu[1] = {10000};

    const gsl_rng_type *T;
    gsl_rng *r;

    gsl_monte_function G = { &g_mc, 1, &pl };

    size_t calls = 100000;

    gsl_rng_env_setup ();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);


    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (1);

    gsl_monte_vegas_integrate (&G, xl, xu, 1, 10000, r, s,
                               &res, &err);
    display_results ("vegas warm-up", res, err);

    printf ("converging...\n");

    do
    {
        gsl_monte_vegas_integrate (&G, xl, xu, 1, calls/5, r, s,
                                   &res, &err);
        std::cout << "result = " << res << " sigma = " << err <<
                  " chisq/dof = " << gsl_monte_vegas_chisq (s) << std::endl;
    }
    while (fabs (gsl_monte_vegas_chisq (s) - 1.0) > 0.5);

    display_results ("vegas final", res, err);

    gsl_monte_vegas_free (s);
    gsl_rng_free (r);

    return res;
}
