from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

import GPy
#GPy.plotting.change_plotting_library('plotly')

meanf = lambda x : 20*np.abs(np.sin(x) / (x + 10))

scale = 1.0


def random_dist(x, n):
    shape = x/scale
    return np.random.gamma(shape, scale, (n,))


def draw_mean(minx, maxx):
    x = np.linspace(minx, maxx, (maxx-minx)*50)
    f = meanf(x)
    return x,f


def draw_points(minx, maxx, nbpoints=100, replicate=1):
    x = minx + (maxx - minx) * np.random.random_sample((nbpoints, ))
    X = np.zeros((nbpoints*replicate,))
    Y = np.zeros((nbpoints*replicate,))
    index = 0
    for el in x:
        m = meanf(el)
        v = random_dist(m, replicate)
        X[index:index+replicate] = el
        Y[index:index+replicate] = v
        index += replicate
    return X, Y


minx = -6
maxx = 6

v,f = draw_mean(minx, maxx)

X, Y = draw_points(minx, maxx, 50, 1)

plt.plot(v,f)
plt.plot(X, Y, '+r')
plt.show()

### Estimate GPs
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
kernel = GPy.kern.RBF(input_dim=1, variance=0.001, lengthscale=1.)
m = GPy.models.GPRegression(X, Y, kernel)
m.optimize(messages=True)
m.optimize_restarts(num_restarts=5)
print(m)
p = np.array([-4.7]).reshape(-1,1)
mu, var = m.predict(p, full_cov=True)
print(p, mu, np.sqrt(var))
#p[0,0] = 0
mu, var = m.predict(p)
print(p, mu, np.sqrt(var))
fig = m.plot()
GPy.plotting.show(fig)
plt.show()

###
