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


# -------------------- utilities ----------------------------------- #


def scatter_plot(x, y):
    plt.figure(figsize=(10, 10))
    for idx, x_i in enumerate(x.T):
        plt.subplot(2, 2, idx+1)
        plt.scatter(x_i, y)
        plt.xlabel(f'x_{idx+1}')
        plt.ylabel(f'y', rotation=0)
    plt.subplot(2, 2, idx+2)
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel(f'x_{idx}')
    plt.ylabel(f'x_{idx+1}', rotation=0)

# -------------------- simulate random data -------------------------- # 

# set the random seed
np.random.seed(42)
# set the number of draws 
N = 100
# set the confounder
x_1 = np.random.normal(size = N)
# set the independent observed variable 
x_2 = z + np.random.normal(size = N, scale = 0.01)
# set the y variable 
y = z + np.random.normal(size = N)
# stack the confounder and the observed in a matrix
X = np.stack((x_1,x_2)).T
scatter_plot(X,y)

# -------------------- specify a full model -------------------------- # 

with pm.Model() as model:
    # set the priors
    alpha = pm.Normal("alpha", mu = 0, sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10, shape = 2)
    sigma = pm.HalfNormal("sigma", 5)
    # set the likelihood 
    y_obs = pm.Normal("y_obs", mu = alpha + pm.math.dot(X,beta), sigma = sigma, observed = y)
    # inference step 
    trace = pm.sample(2000)
    
# ------------------ plot the posterior distribution -------------- # 
    
az.plot_forest(trace, var_names = ["beta"], figsize = (8,2), combined = True)
az.plot_joint(trace, var_names = ["beta"],kind = "kde")
az.plot_pair(trace, var_names=['beta'])
