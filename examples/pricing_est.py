import numpy as np
from time import time
from optparse import OptionParser
from scipy.stats import gamma, norm
import matplotlib.pylab as plt
import GPy
from sklearn import mixture

gmm = mixture.GMM(n_components=3, n_iter=1)
gmm.means_ = np.array([[2],[4], [8]])
gmm.covars_ = np.array([[0.1], [0.1], [0.3]]) ** 2
gmm.weights_ = np.array([0.6, 0.1, 0.3])

# MU = [2;4;8];
# SIGMA = cat(3,[0.1]^2,[0.1]^2,[0.1]^2);
# p = [0.6,0.1,0.3];
# obj = gmdistribution(MU,SIGMA,p);
# plot(0:0.1:10, [0:0.1:10]' .* (1- cdf(obj,[0:0.1:10]')))

import os
import sys
sys.path.append(os.path.abspath('../'))
import maxest.estimate as es
import maxest.fixsestimate as fixes
from time import time

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

# parse commandline arguments
op = OptionParser()
op.add_option("--folder", default='pricing_results',
              dest="folder", type="str",
              help="Destination folder.")
op.add_option("--nbins", default=10,
              dest="nbins", type="int",
              help="Number of bins.")
op.add_option("--nsamples", default=300,
              dest="nsamples", type="int",
              help="Number of samples.")
op.add_option("--suffix", 
              dest="suffix", type="str",
              help="Suffix for data name.")
op.add_option("--clean_folder",
              dest="clean_folder", default=False,
              help="Delete the folder storing results if it exists")
op.add_option("--exclude_weighted",
              dest="exclude_weighted", default=False,
              help="Exclude weighted estimator")


# print(__doc__)
# op.print_help()

(opts, args) = op.parse_args()

folder = opts.folder
suffix = opts.suffix
nbins = opts.nbins
nsamples = opts.nsamples

directory = os.path.abspath(folder)
if not os.path.exists(directory):
    os.makedirs(directory)
elif opts.clean_folder:
    import shutil
    shutil.rmtree(directory)
    os.makedirs(directory)


mu = 3
sigma = 1

#range
minPrice = 0
maxPrice = 10

#gamma Parameter
shape = 2
scale = 1.5


#DRAW SAMPLES
# tau = np.random.gamma(shape, scale, nsamples)
tau = gmm.sample(nsamples).ravel()
actions = np.random.rand(nsamples)*(maxPrice-minPrice) + minPrice

actionsBins = np.linspace(minPrice, maxPrice, nbins+1)
discreteActions=np.digitize(actions,actionsBins)

rewardsW=np.zeros(actions.size)
rewardsW[actions<=tau]=actions[actions<=tau]
# rewardsDiscrete=np.zeros(actions.size);
# rewardsDiscrete[discreteActions<=tau]=discreteActions[discreteActions<=tau];

data = np.hstack((tau,actions,rewardsW))
path = os.path.join(directory, 'data_'+str(nbins)+'_'+str(nsamples)+'_'+suffix+'.csv')
np.savetxt(path, data, delimiter=',')

# plt.scatter(actions, rewardsW)
# plt.show()
# exit(8)


###############################################################################
# REAL MAX
###############################################################################
# x=np.linspace(0,maxPrice,1000)
# # a=gamma.pdf(x,shape, scale=scale)
# a=norm.pdf(x,8,0.1)
# # p = x*(1 - gamma.cdf(x, a=shape, scale=scale))
# p = x*(1 - norm.cdf(x, 8, 0.1))
#  
#  
# plt.plot(x,a)
# plt.plot(x,p)
# plt.show()



###############################################################################
# Maximum ME
###############################################################################
means=np.zeros(nbins)
for i in range(1,nbins+1):
    means[i-1] = rewardsW[discreteActions == i].mean() 
    if i not in discreteActions:
        means[i-1]=0.0

maxMeanME = max(means);
print('MME maximum: {}'.format(maxMeanME))

path_name = os.path.join(directory, 'MME_'+str(nbins)+'_'+str(nsamples)+'.txt')
with open(path_name, "a+") as myfile:
    myfile.write(suffix + ',' + str(maxMeanME) + '\n')

###############################################################################
# Maximum Double Estimator
###############################################################################
half_batch = int(len(rewardsW)/2.0)
rewardsA = rewardsW[:half_batch]
rewardsB = rewardsW[half_batch:]
actionsA = discreteActions[:half_batch]
actionsB = discreteActions[half_batch:]
assert(len(rewardsA) == len(actionsA))
assert(len(rewardsB) == len(actionsB))

meansA = np.zeros(nbins);
for i in range(1, nbins+1):
    meansA[i-1] = rewardsA[actionsA == i].mean() 
    if i not in actionsA:
        meansA[i-1] = 0.0;

meansB = np.zeros(nbins);
for i in range(1, nbins+1):
    meansB[i-1] = rewardsB[actionsB == i].mean() 
    if i not in actionsB:
        meansB[i-1] = 0.0;

maxActionA = np.argmax(meansA)
maxActionB = np.argmax(meansB)

maxMeanDouble = (meansA[maxActionB] + meansB[maxActionA])/2.0
print('MMD maximum: {}'.format(maxMeanDouble))

path_name = os.path.join(directory, 'MMD_'+str(nbins)+'_'+str(nsamples)+'.txt')
with open(path_name, "a+") as myfile:
    myfile.write(suffix + ',' + str(maxMeanDouble) + '\n')

###############################################################################
# Maximum Probability
###############################################################################
if not opts.exclude_weighted:
    x = actions.reshape(-1,1);
    y = rewardsW.reshape(-1,1);
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=0.01);
    gp = GPy.models.GPRegression(x,y,kernel);
    gp.optimize_restarts(num_restarts=8, verbose=False)
    print(gp)
    
#     fig = gp.plot()
#     GPy.plotting.show(fig)
#     plt.show()
    
    #Maximum weighted
#     maximumWE, h = es.predict_max(gp, minPrice, maxPrice, verbose=1,
#                                 tfinp=es.scalar2array, epsabs=0.01, epsrel=0.01, limit=30)
    maximumWE = fixes.compute_max(gp, minPrice, maxPrice, es.scalar2array)
    print('MWE maximum: {}'.format(maximumWE))
      
    path_name = os.path.join(directory, 'MWE_'+str(nbins)+'_'+str(nsamples)+'.txt')
    with open(path_name, "a+") as myfile:
        myfile.write(suffix + ',' + str(maximumWE) + '\n')
        
#     start = time()
#     print("val: ", es.compute_max(gp, minPrice, maxPrice, tfinp=es.scalar2array, ops={"epsabs":1.49e-06, "epsrel":1.49e-06, "limit":10}))
#     print("t: ", time()-start)

        
    start = time()
    print("val: ", fixes.compute_max(gp, minPrice, maxPrice, es.scalar2array))
    print("t: ", time()-start)
    
