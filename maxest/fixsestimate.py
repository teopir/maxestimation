from __future__ import print_function
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from time import time
import numbers
from math import log, exp
from .estimate import product_integral, array2scalar, prob_z_is_max, prod_int_div_cdf

from joblib import Parallel, delayed

import logging
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def aaaa(z, gps, minz, maxz, verbose=0, tfinp=lambda x:x):
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
    ins = sigma * np.random.randn(50) + mu
    val = 0
    for i in range(50):
        val = val + myf(ins[i])
    val = val / 50
    if verbose > 0:
        print('t[expected]: {}, #p: {}'.format(time()-start, nump))
    return val


def tmpf(gps, x, minz, maxz, tfinp):
#         print('x:', x)
    mu, _ = gps.predict(tfinp(x))
    mu = array2scalar(mu)
    assert isinstance(mu, numbers.Number)
#     v = prob_z_is_max(x, gps, minz, maxz, 1, tfinp)
    v = aaaa(x, gps, minz, maxz, 1, tfinp)
    value = mu * v
    return value

def compute_max(gps, minz, maxz, tfinp, n_jobs=-1):
    
    val_iterable = np.linspace(minz, maxz, (maxz-minz) * 10)
    
    out = Parallel(
            n_jobs=n_jobs, verbose=1
            )(
              delayed(tmpf)(gps, value, minz, maxz, tfinp)
              for value in val_iterable)            
    print(out)
    
    max_est = integrate.simps(out, val_iterable)
    return max_est
    

    