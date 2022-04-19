#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: econeutics

"""

import numpy as np 
import pandas as pd 
import logging
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
x_2 = z + np.random.normal(size = N, scale = 1)
# set the y variable 
y = z + np.random.normal(size = N)
# stack the confounder and the observed in a matrix
X = np.stack((x_1,x_2)).T
scatter_plot(X,y)

# -------------------- model of y on x_1 and x_2 -------------------------- # 

with pm.Model() as m_x1x2:
    # set the priors on the parameters
    alpha = pm.Normal("alpha",mu = 0, sd = 10)
    beta_1 = pm.Normal("beta 1",mu = 0, sd = 10)
    beta_2 = pm.Normal("beta 2",mu = 0, sd = 10)
    sigma = pm.HalfNormal("sigma", 5)
    #model the mean of the data
    mu = pm.Deterministic("mu", alpha + beta_1 * X[:,0] + beta_2 * X[:,1])
    # set the likelihood
    y_obs = pm.Normal("y_obs", mu = mu, sigma = sigma, observed = y)
    # inference step
    trace_x1x2 = pm.sample(2000)

# -------------------- model of y on x_1 --------------------------------- # 
    
with pm.Model() as m_x1:
    # set the priors on the parameters
    alpha = pm.Normal("alpha",mu = 0, sd = 10)
    beta_1 = pm.Normal("beta 1",mu = 0, sd = 10)
    sigma = pm.HalfNormal("sigma", 5)
    #model the mean of the data
    mu = pm.Deterministic("mu", alpha + beta_1 * x_1)
    # set the likelihood
    y_obs = pm.Normal("y_obs", mu = mu, sigma = sigma, observed = y)
    # inference step
    trace_x1 = pm.sample(2000)
    
# -------------------- model of y on x_2 ----------------------------------- #

with pm.Model() as m_x2:
    # set the priors on the parameters
    alpha = pm.Normal("alpha",mu = 0, sd = 10)
    beta_2 = pm.Normal("beta 2",mu = 0, sd = 10)
    sigma = pm.HalfNormal("sigma", 5)
    #model the mean of the data
    mu = pm.Deterministic("mu", alpha + beta_2 * x_2)
    # set the likelihood
    y_obs = pm.Normal("y_obs", mu = mu, sigma = sigma, observed = y)
    # inference step
    trace_x2 = pm.sample(2000)
    
# --------------------- analyse the posterior ------------------------------- # 

az.plot_forest([trace_x1x2,trace_x1,trace_x2], model_names = ["m_x1x2", "m_x1", "m_x2"], var_names = ["beta 1","beta 2"], combined = False, colors = "cycle", figsize = (8,3))
