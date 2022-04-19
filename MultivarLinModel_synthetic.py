#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import logging 
import numpy as np 
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import pymc3 as pm 
import arviz as az

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')
# -------------------------- generate syntethic data ---------------------------------------------- # 

# set the random seed
np.random.seed(314)
# set the number of draws
N = 100
# set the true values for the constant and coefficients
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = ss.norm.rvs(loc = 0, scale = 1, size = N)
# define the X variable
X = np.random.normal(loc = [10,5], scale = [10,5], size = (N,2))
log.info("The original X variable is: %s", X)
# center the X variable
X_centered = X - X.mean(0)
log.info("The centered X variable is: %s", X_centered)
# get the y variable 
y = alpha_real + np.dot(X, beta_real) + eps_real
log.info("The y variable is: %s", y)

# -------------------------- specify a probabilistic model ----------------------------------- # 

with pm.Model() as multiv_model:
    # set the parameters priors
    alpha_temp = pm.Normal("alpha_temp", mu = y.mean(), sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10, shape = 2)
    sigma = pm.HalfCauchy("sigma", 5)
    # specify the mean of the data 
    mu = alpha_temp + pm.math.dot(X_centered, beta)
    # get the original alpha
    alpha = pm.Deterministic("alpha", alpha_temp - pm.math.dot(X.mean(0), beta))
    # get the likelihood 
    y_obs = pm.Normal("y_obs", mu = mu, sigma = sigma, observed = y)
    trace= pm.sample(2000)
    log.info("the inference summary is: %s",az.summary(trace))
    

