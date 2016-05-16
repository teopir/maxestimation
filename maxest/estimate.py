import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

def product_integral(f, a, b, verbose=0):
    """
    \exp(\int_a^b \ln f(x) dx)
    :param f: the function to be evaluated
    :param a: the lower bound of the definite integral
    :param b: the upper bound of the definite integral
    :return: the product integral
    """
    result = integrate.quad(lambda x: np.log(f(x)), a, b)
    expres = np.exp(result[0])
    return expres


def prod_int_div_cdf(x, mu, sigma, gps, miny, maxy):
    """
    ..math:
        \frac{
        \Prod_{miny}^{maxy} CDF[x, \mathcal{N}(\mu^{GP}(y), \sigma^{GP}(y))]^{dy}
        }{
        CDF[x, \mathcal{N}(\mu, \sigma)]
        }

    :param x:
    :param mean:
    :param std:
    :param gps:
    :param miny:
    :param maxy:
    :return:
    """
    def fpi(y):
        loc, scale = gps.predict(y)
        return norm.cdf(x, loc=loc, scale=scale)

    # compute comulative distribution
    cdfx = norm.cdf(x, loc=mu, scale=sigma)

    # compute product integral
    pi = product_integral(fpi, miny, maxy)

    # compute product_integral/cumulative_function
    if np.isnan(pi/cdfx):
        return 0.0
    return pi / cdfx


def prob_z_is_max(z, gps, minz, maxz):
    assert minz <= z <= maxz, '{} not in [{}, {}]'.format(z, minz, maxz)
    mu, sigma = gps.predict(z)
    
    def myf(x):
        return prod_int_div_cdf(x,mu, sigma, gps, minz, maxz)

    return norm.expect(myf,
                loc=mu, scale=sigma)

def predict_max(gps, minz, maxz):

    def tmpf(x):
        mu, _ = gps.predict(x)
        v = prob_z_is_max(x, gps, minz, maxz)
        value = mu * v
        return value

    return integrate.quad(tmpf, minz, maxz)[0]

if __name__ == "__main__":
    mean = 1.0
    stddev = 2.0
    f = lambda x: norm.cdf(x, mean, stddev)
    print(f(3))
    v = product_integral(f, -2, 5, 1)