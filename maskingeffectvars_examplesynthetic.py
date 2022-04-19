#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:09:38 2020

@author: davideferri
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

# ------------------------------ simulate data ----------------------------------- # 

# set the random seed
np.random.seed(42)
# set the number of trials 
N = 126
# set the correlation coefficient
r = 0.8
# define the independent variables
x_1 = np.random.normal(size = N)
x_2 = np.random.normal(x_1, scale = (1 - r**2)**0.5)
# define an independent variable 
y = np.random.normal(x_1 - x_2)
# stack the independent variables
X = np.vstack((x_1,x_2)).T
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

az.plot_forest([trace_x1x2,trace_x1,trace_x2], model_names = ["m_x1x2", "m_x1", "m_x2"], var_names = ["beta 1","beta 2"], combined = True, colors = "cycle", figsize = (8,3))
