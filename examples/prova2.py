import numpy as np
from time import time

import os
import sys
sys.path.append(os.path.abspath('../'))

import maxest.estimate as es
import maxest.fixsestimate as fixes

class test2:
    def predict(self, x):
        #print(x)
        return np.array([x*(1-x)]), np.array([0.1])
    
    def ess(self, x):
        return 1.0
    
start = time()
val = test2()
print(es.predict_max(val, 0, 1, verbose=0, epsabs=1.49e-06, epsrel=1.49e-06, limit=50, tfinp=es.scalar2array))
# print(es.predict_max(val, 0, 1, verbose=0, vec_func=False, tol=1.49e-02, rtol=1.49e-02))
print("t: ", time()-start,'s')

start = time()
print("val: ", es.compute_max(val, 0, 1, tfinp=es.scalar2array, ops={"epsabs":1.49e-06, "epsrel":1.49e-06, "limit":50}))
print("t: ", time()-start)

start = time()
print("val: ", fixes.compute_max(val, 0, 1, es.identityfunc))
print("t: ", time()-start)
