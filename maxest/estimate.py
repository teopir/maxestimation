import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from time import time
import numbers

def array2scalar(x):
    if isinstance(x, np.ndarray):
        assert len(x) == 1, 'Expected array of dimension 1. Given {}'.format(x)
        return np.asscalar(x)
    elif isinstance(x, list):
        assert len(x) == 1, 'Expected array of dimension 1. Given {}'.format(x)
        return x[0]
    elif isinstance(x, numbers.Number):
        return x
    else:
        raise ValueError('Unknown type supported (list, np.ndarray, numbers)')

def scalar2array(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1,1)
    elif isinstance(x, numbers.Number):
        return np.array([[x]])
    else:
        raise ValueError('unknown type')

def product_integral(f, a, b, verbose=0):
    """
    \exp(\int_a^b \ln f(x) dx)
    :param f: the function to be evaluated
    :param a: the lower bound of the definite integral
    :param b: the upper bound of the definite integral
    :return: the product integral
    """
    result = integrate.quad(lambda x: np.log(f(x)), a, b, epsabs=0.001, epsrel=0.001)
    expres = np.exp(result[0])
    return expres


def prod_int_div_cdf(x, mu, sigma, gps, miny, maxy, verbose=0, tfinp=lambda x:x):
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
        loc, var = gps.predict(tfinp(y))
        ess = gps.ess(tfinp(y))
        loc, var = array2scalar(loc), array2scalar(var)
        scale = np.sqrt(var/ess)
        return norm.cdf(x, loc=loc, scale=scale)

    # compute comulative distribution
    cdfx = norm.cdf(x, loc=mu, scale=sigma)

    # compute product integral
    start = time()
    pi = product_integral(fpi, miny, maxy)
    if verbose > 0:
        print('t[prod_int]: {}'.format(time()-start))

    # compute product_integral/cumulative_function
    if np.isnan(pi/cdfx):
        return 0.0
    return pi / cdfx

nump = 0

def prob_z_is_max(z, gps, minz, maxz, verbose=0, tfinp=lambda x:x):
    assert minz <= z <= maxz, '{} not in [{}, {}]'.format(z, minz, maxz)
    mu, var = gps.predict(tfinp(z))
    ess = gps.ess(tfinp(z))
    print(z, ess)
    mu, var = array2scalar(mu), array2scalar(var)
    sigma = np.sqrt(var/ess)
    
    global nump
    nump = 0
    
    def myf(x):
        global nump 
        nump += 1
        return prod_int_div_cdf(x, mu, sigma, gps, minz, maxz, verbose=0, tfinp=tfinp)

    start = time()
    
    val = norm.expect(myf, loc=mu, scale=sigma, lb=mu-3.5*sigma, ub=mu+3.5*sigma,
                      epsabs=0.001, epsrel=0.001)
    if verbose > 0:
        print('t[expected]: {}, #p: {}'.format(time()-start, nump))
    return val

def predict_max(gps, minz, maxz, verbose=0, tfinp=lambda x:x, **kwargs):
    
    history = []

    def tmpf(x):
#         print('x:', x)
        mu, _ = gps.predict(tfinp(x))
        mu = array2scalar(mu)
        assert isinstance(mu, numbers.Number)
        v = prob_z_is_max(x, gps, minz, maxz, verbose=verbose, tfinp=tfinp)
        value = mu * v
        history.append([x,v])
        assert isinstance(value, numbers.Number), '{}'.format(value)
        return value

    return integrate.quad(tmpf, minz, maxz, **kwargs)[0], history

if __name__ == "__main__":
    mean = 1.0
    stddev = 2.0
    f = lambda x: norm.cdf(x, mean, stddev)
    print(f(3))
    v = product_integral(f, -2, 5, 1)