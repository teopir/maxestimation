import GPy
import numpy as np
from math import sqrt

kernel = GPy.kern.RBF(input_dim=1, variance=0.000926064771899546*0.000926064771899546, lengthscale=0.001000000000000000)

X = np.loadtxt('/home/matteo/Projects/maxest/c++/build-maxest-Desktop-Default/examples/x.dat')
X = X.reshape(-1,1)
Y = np.loadtxt('/home/matteo/Projects/maxest/c++/build-maxest-Desktop-Default/examples/y.dat')
Y = Y.reshape(-1,1)

m = GPy.models.GPRegression(X,Y,kernel,
                            noise_var=1.249841256846034643*1.249841256846034643)
print(m)

Xnew = np.array([2]).reshape(-1,1)
loc, var = m.predict(Xnew)

print(loc)
print(sqrt(var))
