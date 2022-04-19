#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:57:11 2020

@author: davideferri
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

# ---------------------- specify the pymc3 model ---------------------------------- # 

with pm.Model() as linear_model:
    # set the priors
    beta = pm.Normal("beta",0,10)
    alpha = pm.Normal("alpha",0,10)
    sigma = pm.HalfNormal("sigma",10)
    mu = pm.Deterministic("mu",alpha + beta * x)
    # set the likelihood of the observations
    obs = pm.Normal("obs", mu = mu, sigma = sigma, observed = y)
    # inference step 
    trace = pm.sample(2000,tune = 1000)
    
# # ------------------- analyse the posterior ------------------------------------- # 
    
with linear_model:
    log.info("the trace summary is: %s", az.summary(trace))
    #plt the results
    az.plot_joint(trace, kind = "kde", var_names = ["beta","alpha"])
    az.plot_trace(trace)
    az.plot_posterior(trace, var_names = ["alpha","beta"], rope = [-0.05,0.05], credible_interval = 0.9)
    
# -------------------- interpreting and visualizing the posterior ----------------- # 

# plot 1
# replot the true data 
fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(x,y)
# get the mean of the parameters
alpha_m = trace["alpha"].mean()
beta_m = trace["beta"].mean()
# take some draws from the posterior of parameters
draws = range(0,len(trace["alpha"]),10)
ax.plot(x,trace["alpha"][draws] + trace["beta"][draws] * x[:,np.newaxis], c = "gray", alpha = 0.5)
ax.plot(x, alpha_m + beta_m * x, c = "k",label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
ax.set_xlabel("x")
ax.set_ylabel("y",rotation = 0)
plt.legend()
plt.show()
# plot 2 
fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(x,y)
ax.plot(x, alpha_m + beta_m * x, c = "k",label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hpd(x,trace["mu"],credible_interval = 0.98,color = "k")
ax.set_xlabel("x")
ax.set_ylabel("y",rotation = 0)
plt.legend()
plt.show()
# plot 3
# get the y from the posterior distribution 
ppc = pm.sample_posterior_predictive(trace,samples = 2000,model = linear_model)
plt.figure()
plt.plot(x,y,"b.")
plt.plot(x,alpha_m + beta_m * x,c = "k",label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hpd(x,ppc["obs"],credible_interval = 0.5, color = "gray")
az.plot_hpd(x,ppc["obs"],color = "gray")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ------------------- get the R squared of our model ------------------------ # 

r_squared = az.r2_score(y, ppc["obs"])
log.info("The r squared of the model is: %s", r_squared)


    