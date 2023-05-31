#-------------------------------------------------------------------------#
#                   Gibbs Sampling with Data Augmentation Algorithm
#                             Author: Keegan Skeate
#                       Copyright 2016. All Rights Reserved.
#-------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import scipy.stats as ss
from math import *
from gibbs_probit_graphs import *
#-------------------------------------------------------------------------#
def gibbs(beta0,B0,v0,d0,y,X,iters):
    """
    Inputs:
        Priors - beta0, B0, v0, d0
        Data - y and X
        Desired iterations - iters
    Returns: a sample of parameters from the posterior.
    """
    beta = np.zeros((iters,len(beta0)))
    sigma_sq = np.ones(iters)
    B0_inv = np.linalg.inv(B0)  
    for i in range(iters):
        y_star = y
        XB = np.dot(X,beta[i-1])
        s = sigma_sq[i-1]
        for n in range(len(y)):
            if y[n]==1:
                y_star[n] = ss.truncnorm.rvs(-np.infty,0.,loc=XB[n],scale=s)
            if y[n]==2:
                y_star[n] = ss.truncnorm.rvs(0.,1.,loc=XB[n],scale=s)
            if y[n]==3:
                y_star[n] = ss.truncnorm.rvs(1.,np.infty,loc=XB[n],scale=s)
        v0_hat = v0 + (len(y)/2)
        d0_hat = 1./((1/d0)+.5*(np.dot((y_star-XB).T,y_star-XB)))
        sigma_sq[i] = ss.invgamma.rvs(v0_hat,d0_hat) 
        B_hat = np.linalg.inv(np.dot(X.T,X)/sigma_sq[i]+B0_inv)
        beta_hat = np.dot(B_hat, (np.dot(B0_inv,beta0).T+ \
                   (np.dot(X.T,y_star)/sigma_sq[i])).T)
        beta[i] = np.random.multivariate_normal(np.hstack(beta_hat),B_hat)      
    return beta,sigma_sq



#-------------------------------------------------------------------------#
#                                  Execution
#-------------------------------------------------------------------------#
data = pd.read_excel('gibbs-probit-data.xlsx', header=None)
y = data[0].values
X = data[[1,2,3]].values
# Gibbs Sampling with Augmentation
v0 = 5.0 ; d0 = 100.0
beta0 = np.zeros((3,1))
Sigma0 = 10*identity(3)
iters=11000
beta, sigma_sq = gibbs(beta0,Sigma0,v0,d0,y,X,iters)
# 10% Burn-In
model_parameters = np.column_stack([beta[0.1*iterations:],
                                    sigma_sq[0.1*iterations:]])
# Posterior means and standard deviations
posterior_means = mean(model_parameters, axis=0)
posterior_stdvs = std(model_parameters, axis=0)
print "posterior_means\n",posterior_means
print "posterior_stdvs\n",posterior_stdvs
#------------------------------ Figures ----------------------------------#
posterior_fig = fig(model_parameters)
posterior_fig.savefig('posterior_fig.png', format='png', dpi=300,bbox_inches='tight')