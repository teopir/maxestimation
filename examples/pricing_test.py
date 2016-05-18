import numpy as np
from time import time

import os
import sys
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
from scipy.stats import gamma


import maxest.estimate as es
import GPy


def ess(self, Xnew, kern=None):
	kern = self.kern if kern is None else kern
	X = self._predictive_variable
	Winv = self.posterior.woodbury_inv
	Kx = kern.K(X, Xnew)
	weights = np.dot(Kx.T, Winv)
	return 1.0 / np.asscalar(np.dot(weights,weights.T))

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








#Maximum ME
means=np.zeros(nbins);
for i in range(1,nbins+1):
	means[i-1]=rewardsW[discreteActions == i].mean() 
	if(i not in discreteActions):
		means[i-1]=0.0;


maxActionME=np.argmax(means);
maxMeanME=max(means);

#print(tau);
#print(actionsBins);
#print(discreteActions);
#print(rewardsDiscrete);
print(means);

print("max action is :",maxActionME);
print("maxAction_value:", (actionsBins[maxActionME] + actionsBins[maxActionME-1])/2.0)
print("with value :",maxMeanME);



#Maximum double estimator
rewardsA=rewardsW[1:len(rewardsW)/2];
rewardsB=rewardsW[len(rewardsW)/2 + 1:len(rewardsW)];
actionsA=discreteActions[1:len(discreteActions)/2];
actionsB=discreteActions[len(discreteActions)/2 + 1 :len(discreteActions)];




meansA=np.zeros(nbins);
for i in range(1,nbins+1):
	meansA[i-1]=rewardsA[actionsA == i].mean() 
	if(i not in actionsA):
		meansA[i-1]=0.0;

meansB=np.zeros(nbins);
for i in range(1,nbins+1):
	meansB[i-1]=rewardsB[actionsB == i].mean() 
	if(i not in actionsB):
		meansB[i-1]=0.0;

maxActionA=np.argmax(meansA);
maxActionB=np.argmax(meansB);

maxActionDouble = ( maxActionA + maxActionB )/2.0;
maxMeanDouble=(meansA[maxActionB] + meansB[maxActionA])/2.0;

print("maxActionDouble:", maxActionDouble)
print("maxActionDouble_value:", (actionsBins[maxActionDouble] + actionsBins[maxActionDouble-1])/2.0)
print("maxMeanDouble:",maxMeanDouble);





#GP

x = actions.reshape(-1,1);
y = rewardsW.reshape(-1,1);
kernel = GPy.kern.RBF(input_dim=1, variance=0.01, lengthscale=0.00001);
gp = GPy.models.GPRegression(x,y,kernel);
print(gp)
gp.optimize_restarts(num_restarts=8)
print(gp)

posterior = gp.posterior
winv = posterior.woodbury_inv
K = posterior._K
prod = np.dot(np.atleast_3d(winv).T, K)
ess_inv = np.dot(prod, prod)

fig = gp.plot()
GPy.plotting.show(fig)
plt.show()





#Maximum weighted
maximumWE, h = es.predict_max(gp, minPrice, maxPrice, verbose=1,
							tfinp=es.scalar2array, epsabs=0.001, epsrel=0.001, limit=30)

print(h)

act = 0
for el in h:
	act += el[0] * el[1]
print(maximumWE)
print(act)











