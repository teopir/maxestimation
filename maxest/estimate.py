from __future__ import print_function
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from time import time
import numbers
from math import log, exp


import logging
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def identityfunc(x):
    return x
    
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
#     result = integrate.quad(lambda x: log(f(x)), a, b, epsabs=0.001, epsrel=0.001)
#     expres = exp(result[0])
    result = integrate.romberg(lambda x: log(f(x)), a, b, tol=0.00001, rtol=0.00001)
    expres = exp(result)
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
    #print(z, ess)
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

def compute_max(gps, minz, maxz, tfinp=lambda x: x, ops={}):
    
    def bound_x(z):
        # print("z: ", z)
        z_tr = tfinp(z)
        mu_z, var_z = gps.predict(z_tr)
        ess = gps.ess(z_tr)
        mu_z, var_z = array2scalar(mu_z), array2scalar(var_z)
        # compute standard deviation with central limit theorem
        sigma_clt_z = np.sqrt(var_z/ess)
        # print(mu_z - 4 * sigma_clt_z, mu_z + 4 * sigma_clt_z)
        return [mu_z - 3.5*sigma_clt_z, mu_z + 3.5*sigma_clt_z]
    
    def prod_int_funct(x, y):
        y_tr = tfinp(y)
        mu_y, var_y = gps.predict(y_tr)
        ess = gps.ess(y_tr)
        mu_y, var_y = array2scalar(mu_y), array2scalar(var_y)
        scale = np.sqrt(var_y/ess)
        return norm.cdf(x, loc=mu_y, scale=scale)
    
    def inner_f(x, z):
        #print("x: {}, z: {}".format(x, z))
        start = time()
        z_tr = tfinp(z)
        mu_z, var_z = gps.predict(z_tr)
        ess = gps.ess(z_tr)
        mu_z, var_z = array2scalar(mu_z), array2scalar(var_z)
#         print('t: ', time()-start)
#         start = time()
        
        # compute standard deviation with central limit theorem
        sigma_clt_z = np.sqrt(var_z/ess)
        fhat_z = norm.pdf(x, loc=mu_z, scale=sigma_clt_z)
        Fhat_z = norm.cdf(x, loc=mu_z, scale=sigma_clt_z)
        
#         print('t: ', time()-start)
#         start = time()
        
        pi_f = lambda y: prod_int_funct(x, y)
        pi = product_integral(pi_f, minz, maxz, 0)
        
        out_val = mu_z * fhat_z * pi / Fhat_z
#        print('t: ', time()-start)
#        print('-'*40)
        return out_val

    # integral ranges are ordered from inner to outer
    return integrate.nquad(inner_f, [bound_x, [minz, maxz]], opts=ops)[0]
        

if __name__ == "__main__":
    mean = 1.0
    stddev = 2.0
    f = lambda x: norm.cdf(x, mean, stddev)
    print(f(3))
    v = product_integral(f, -2, 5, 1)