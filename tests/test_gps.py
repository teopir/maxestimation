import numpy as np
import GPy
import matplotlib.pyplot as plt

def ess(self, Xnew, kern=None):
    kern = self.kern if kern is None else kern
    X = self._predictive_variable
    print(X)
    Winv = self.posterior.woodbury_inv
    print(Winv)
    Kx = kern.K(X, Xnew)
    print(Kx.shape)
    weights = np.dot(Kx.T, Winv)
    print(weights.shape)
    return 1.0 / np.dot(weights,weights.T)

GPy.models.GPRegression.ess = ess

x = np.array([1, 10, 1]).reshape(-1,1)
y = np.array([1,132, 8]).reshape(-1,1)

kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=10000);
gp = GPy.models.GPRegression(x,y,kernel,noise_var=0.0001);
print(gp)
# gp.optimize_restarts(num_restarts=8)
# print(gp)

b = gp.ess(np.array([[1]]))
print(b)
