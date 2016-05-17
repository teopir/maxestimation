import numpy as np
from time import time

import os
import sys
sys.path.append(os.path.abspath('../'))

import maxest.estimate as es

class test2:
    def predict(self, x):
        #print(x)
        return np.array([x*(1-x)]), np.array([0.01])
    
start = time()
val = test2()
print(es.predict_max(val, 0, 1, verbose=1, epsabs=1.49e-01, epsrel=1.49e-01, limit=20, tfinp=es.scalar2array))
# print(es.predict_max(val, 0, 1, verbose=0, vec_func=False, tol=1.49e-02, rtol=1.49e-02))
print(time()-start,'s')