import numpy as np
from time import time

import os
import sys
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
from scipy.stats import gamma


import maxest.estimate as es
import GPy

# Inject effective sample size estimator
def ess(self, Xnew, kern=None):
    kern = self.kern if kern is None else kern
    X = self._predictive_variable
    Winv = self.posterior.woodbury_inv
    Kx = kern.K(X, Xnew)
    weights = np.dot(Kx.T, Winv)
    start = time()
    l1 = np.asscalar(np.linalg.norm(weights.T,1))
    l2 = np.asscalar(np.linalg.norm(weights.T,2))
    #assert(np.allclose(l2*l2, np.asscalar(np.dot(weights,weights.T))))
    return (l1*l1) / (l2*l2)

GPy.models.GPRegression.ess = ess


#Sampling   

#gaussian paramater
mu=3;
sigma=1;

#range
minPrice=0;
maxPrice=10;

#gamma Parameter
shape=2;
scale=1.5

nsamples=300;

#np.random.seed(852952)
tau = np.random.gamma(shape,scale,nsamples)
actions= np.random.rand(nsamples)*(maxPrice-minPrice) + minPrice;

nbins=10;
actionsBins = np.linspace(minPrice, maxPrice, nbins+1);

print(gamma.mean(shape,scale=scale));
print(actionsBins)
discreteActions=np.digitize(actions,actionsBins);
print(actions[:10]);
print(discreteActions[:10])

rewardsW=np.zeros(actions.size);
rewardsDiscrete=np.zeros(actions.size);
rewardsW[actions<=tau]=actions[actions<=tau]; 
rewardsDiscrete[discreteActions<=tau]=discreteActions[discreteActions<=tau];
print(rewardsW);
#plt.scatter(actions,rewardsW)
#plt.show()




# Find real Maximum
x=np.linspace(0,maxPrice,1000);
a=gamma.pdf(x,shape, scale=scale)
p = x*(1 - gamma.cdf(x, a=shape, scale=scale));


plt.plot(x,a);
plt.plot(x,p);
plt.show();

x = actions.reshape(-1,1);
y = rewardsW.reshape(-1,1);
kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=0.01);
gp = GPy.models.GPRegression(x,y,kernel);
gp.optimize_restarts(num_restarts=8, verbose=False)
print(gp)

fig = gp.plot()
GPy.plotting.show(fig)
plt.show()

start=time()

es.product_integral(f, a, b, verbose)

end=time()-start












