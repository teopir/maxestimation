#include "maxest/niestimator.h"


double pdf(double x, double mu, double sigma)
{
    return exp( -1 * (x - mu) * (x - mu) / (2 * sigma * sigma)) / (sigma * sqrt(2 * M_PI));
}

double cdf(double x, double mu, double sigma)
{
    double v = (x - mu) / (sigma * sqrt(2.));
    return 0.5 * (1 + erf(v));
}

double product_integral_exponent (double y, void * params)
{
    MaxEstimatorParameters* p = (MaxEstimatorParameters*) params;
    MaxEstApproximator* qf = p->qf;
    double x = p->x;

    double var, ns;
    double mu = qf->predict(y, var);
    ns = qf->get_noise_sigma();
    double cdf_val = cdf(x, mu, sqrt(var - ns*ns));

    double f = log(fmax(1e-12, cdf_val));
    return f;
}

double prod_int_div_cdf(double x, void * params)
{
    MaxEstimatorParameters* p = (MaxEstimatorParameters*) params;
    MaxEstApproximator* qf = p->qf;

    // Compute product integral
    gsl_integration_cquad_workspace* w = gsl_integration_cquad_workspace_alloc (NBINS);
    gsl_function F;
    F.function = &product_integral_exponent;
    p->x = x;
    F.params = p;
    double result, error;
    size_t intervals;

    gsl_integration_cquad (&F, p->space_lower, p->space_upper, ABS_PREC, REL_PREC,
                           w, &result, &error, &intervals);

    double pi = exp(result);

    if (p->verbose > 1)
    {
        std::cout << "Finished product integral computation..." << std::endl;
        printf ("pi              = % .18f\n", pi);
        printf ("estimated error = % .18f\n", error);
        printf ("intervals       = %zu\n", intervals);
    }

    gsl_integration_cquad_workspace_free (w);

    // Compute CDF
    double mu_z = p->mu;
    double sigma_z = p->sigma;
    double pdfx = pdf(x, mu_z, sigma_z);
    double cdfx = cdf(x, mu_z, sigma_z);

    if (p->verbose > 1)
    {
        printf ("pdfx: %.9f\ncdfx: %.9f\npi: %.9f\n", pdfx, cdfx, pi);
        printf ("E_z: %.18f\n", pdfx*pi/cdfx);
    }

    return pdfx * pi / cdfx;
}

double prob_z_is_max(double z, void* params)
{
    MaxEstimatorParameters* p = (MaxEstimatorParameters*) params;
    MaxEstApproximator* qf = p->qf;


    double var_z, ns;
    double mu_z = qf->predict(z, var_z);
    ns = qf->get_noise_sigma();
    double sigma_z = sqrt(var_z - ns*ns);

    // Compute product integral
    gsl_integration_cquad_workspace* w = gsl_integration_cquad_workspace_alloc (NBINS);
    gsl_function F;
    F.function = &prod_int_div_cdf;
    p->mu = mu_z;
    p->sigma = sigma_z;
    F.params = p;
    double w_z, error;
    size_t intervals;

    gsl_integration_cquad (&F, mu_z-3.5*sigma_z, mu_z+3.5*sigma_z, ABS_PREC, REL_PREC,
                           w, &w_z, &error, &intervals);

    if (p->verbose > 1)
    {
        printf ("Finished expected value computation...\n");
        printf ("w_z             = % .18f\n", w_z);
        printf ("estimated error = % .18f\n", error);
        printf ("intervals       = %zu\n", intervals);
    }

    gsl_integration_cquad_workspace_free (w);

    if (p->verbose > 0)
    {
        printf ("(z, mu_z, w_z) = (%.6f, %.6f, %.6f)\n", z, mu_z, w_z);
    }
    return w_z * mu_z;
}

double ni_predict_max(MaxEstApproximator* qf, double minz, double maxz, int verbose)
{

    // Compute product integral
    gsl_integration_cquad_workspace* w = gsl_integration_cquad_workspace_alloc (NBINS);
    gsl_function F;
    F.function = &prob_z_is_max;
    MaxEstimatorParameters pl;
    pl.qf = qf;
    pl.space_lower = minz;
    pl.space_upper = maxz;
    pl.verbose = verbose;
    F.params = &pl;
    double result, error;
    size_t intervals;

    gsl_integration_cquad (&F, minz, maxz, ABS_PREC, REL_PREC,
                           w, &result, &error, &intervals);

    if (verbose >= 0)
    {
        printf ("result          = % .18f\n", result);
        printf ("estimated error = % .18f\n", error);
        printf ("intervals       = %zu\n", intervals);
    }

    gsl_integration_cquad_workspace_free (w);

    return result;
}
