#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import logging
import numpy as np 
import pandas as pd 
import scipy.stats as ss 
import pymc3 as pm 
import arviz as az

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')
# ---------------------- generate synthetic data --------------------------------------- # 

# number of obs
N = 101
# set the random seed
np.random.seed(123)
# set the independent variable data
x = np.linspace(0,100,N)
# set the value of the true coefficients 
b = 0.35 ; s = 5 ; alpha = 1
# get the errors
err = ss.norm.rvs(loc = 0,scale = s, size = N)
# get the y's
y = alpha + b * x + err
# plot the true data
fig,ax = plt.subplots(1,2,figsize = (8,4))
ax[0].scatter(x,y)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
az.plot_kde(y, ax = ax[1])
ax[1].set_xlabel("y")
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].tick_params(axis='both', which='minor', labelsize=10)
plt.tight_layout()
plt.show()
# stack the data
data = np.stack((x,y)).T
log.info("the data stacked is: %s", data)
log.info("The data stacked shape is: %s",data.shape)

# -------------------- specify the probabilistic model ---------------------------- # 

with pm.Model() as pearson_model:
    # get the prior on the location parameter
    mu = pm.Normal("mu",mu = data.mean(0), sigma = 10, shape = 2)
    # get the priors on the precision parameters
    sigma_1 = pm.HalfNormal("sigma_1", 10)
    sigma_2 = pm.HalfNormal("sigma_2",10)
    # get the prior on the Pearson correlation coefficient
    rho = pm.Uniform("rho", -1, 1)
    # define the resulting R squared
    r_squared = pm.Deterministic("r_squared", rho**2)
    # get the covariance matrix
    cov = pm.math.stack(([sigma_1**2,rho*sigma_1*sigma_2],[rho*sigma_1*sigma_2, sigma_2**2]))
    # get the likelihood 
    obs = pm.MvNormal("obs", mu = mu, cov = cov, observed = data)
    # inference step 
    trace = pm.sample(1000)
    
# -------------------- analyse the posterior ---------------------------------- # 
    
with pearson_model:
    # study the posterior distribution of r squared
    az.plot_trace(trace,var_names = ["r_squared"])
    # get the summary 
    log.info("The trace summary for the r_squared is: %s", az.summary(trace,var_names = ["r_squared"]))