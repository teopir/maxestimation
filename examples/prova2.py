import numpy as np
from time import time

import os
import sys
from pickletools import optimize
sys.path.append(os.path.abspath('../'))

import maxest.estimate as es

class test2:
    def predict(self, x):
        #print(x)
        return x*(1-x), 0.01
    
start = time()
val = test2()
print(es.predict_max(val, 0, 1))
print(time()-start,'s')