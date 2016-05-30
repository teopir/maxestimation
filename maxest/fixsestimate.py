from __future__ import print_function
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from time import time
import numbers
from math import log, exp
from .estimate import product_integral, array2scalar, prob_z_is_max

from joblib import Parallel, delayed

import logging
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def tmpf(gps, x, minz, maxz, tfinp):
#         print('x:', x)
    mu, _ = gps.predict(tfinp(x))
    mu = array2scalar(mu)
    assert isinstance(mu, numbers.Number)
    v = prob_z_is_max(x, gps, minz, maxz, 1, tfinp)
    value = mu * v
    return value

def compute_max(gps, minz, maxz, tfinp, n_jobs=-1):
    
    val_iterable = np.linspace(minz, maxz, (maxz-minz) * 20)
    
    out = Parallel(
            n_jobs=n_jobs, verbose=1
            )(
              delayed(tmpf)(gps, value, minz, maxz, tfinp)
              for value in val_iterable)            
    print(out)
    
    max_est = integrate.simps(out, val_iterable)
    return max_est
    

    