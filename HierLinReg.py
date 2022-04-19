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

# --------------------- generate synthetic data -------------------------- # 

# get the number of observations by group
N = 20 
# get the number of groups 
M = 8
# define the index array; all group have N observations but the last (only 1)
idx = np.repeat(range(M-1),N)
idx = np.append(idx,7)
log.info("The index list is: %s",idx)
# set a random seed 
np.random.seed(314)

# define the real coefficients
alpha_real = ss.norm.rvs(loc=2.5,scale=0.5,size=M)
log.info("The alpha real is: %s", alpha_real)
beta_real = np.random.beta(6,1,size=M)
log.info("The beta real is: %s", beta_real)
eps_real = np.random.normal(0,0.5,size=len(idx))

# set the independent variable
x_m = np.random.normal(10,1,len(idx))
# set the dependent variable
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real
# plot the true data
fig,ax = plt.subplots(2,4, figsize = (10,5), sharex = True, sharey = True)
ax = np.ravel(ax)
# initialize j and k
j, k = 0, N
for i in range(M):
    # scatter the data
    ax[i].scatter(x_m[j:k],y_m[j:k])
    # set the x label
    ax[i].set_xlabel(f"x_{i}")
    # set the y label
    ax[i].set_ylabel(f"y_{i}",rotation = 0, labelpad = 15)
    # set the x axis limit
    ax[i].set_xlim(6,15)
    # set the y axis limit
    ax[i].set_ylim(7,17)
    # update j,k 
    j += N
    k += N
plt.tight_layout()
plt.show()
# let us center the x data 
x_centered = x_m - x_m.mean()
    
# --------------- specify a non-hierarchical (unpooled) probabilistic model -------------------------- #

with pm.Model() as unpooled_model:
     # set the priors on parameters
     alpha_temp = pm.Normal("alpha_temp", mu = 0, sd = 10, shape = M)
     beta = pm.Normal("beta",mu = 0, sd = 10, shape = M)
     # get the alpha for the uncentered data
     alpha = pm.Deterministic("alpha", alpha_temp - beta * x_m.mean())
     # set the priors on scale and df
     sigma = pm.HalfCauchy("sigma",5)
     df = pm.Exponential("df",1/30)
     # specify the likelihood of the data
     y_obs = pm.StudentT("y_obs", mu = alpha_temp[idx] + beta[idx] * x_centered, sd = sigma, nu = df, observed = y_m)
     # inference step 
     trace_unp = pm.sample(2000)

# -------------- analyse the posterior -------------------------------------- # 
     
with unpooled_model:
    az.plot_forest(trace_unp, var_names = ["alpha","beta"], combined = True)
    
# ---------------- specify a hierarchical probabilistic model ----------------------------- #
    
with pm.Model() as hierarchical_model:
    # specify a set of hyper-priors
    alpha_m_temp = pm.Normal("alpha_m_temp", mu = 0, sd = 10)
    alpha_s_temp = pm.HalfNormal("alpha_s_temp",sd = 10)
    beta_m = pm.Normal("beta_m", mu = 0, sd = 10)
    beta_s = pm.HalfNormal("beta_s",sd = 10)
    # set the priors on parameters
    alpha_temp = pm.Normal("alpha_temp", mu = alpha_m_temp, sd = alpha_s_temp, shape = M)
    beta = pm.Normal("beta",mu = beta_m, sd = beta_s, shape = M)
    # get the alpha for the uncentered data
    alpha = pm.Deterministic("alpha", alpha_temp - beta * x_m.mean())
    alpha_m = pm.Deterministic("alpha_m", alpha_m_temp - beta_m * x_m.mean())
    # set the priors on scale and df
    sigma = pm.HalfCauchy("sigma",5)
    df = pm.Exponential("df",1/30)
    # set the likelihood 
    y_obs = pm.StudentT("y_obs", mu = alpha_temp[idx] + beta[idx] * x_centered, sd = sigma, nu = df, observed = y_m)
    # inference step 
    trace_hm = pm.sample(2000,tune = 2000)
     
# -------------- analyse the posterior ------------------------------  #
     
with hierarchical_model:
    az.plot_forest(trace_hm, var_names = ["alpha","beta"], combined = True)
    az.plot_trace(trace_hm, var_names = ["beta_m","alpha_m"])
    
# # ----------------- plot the regression results for each one of the models ------------------------  #
    
fig,ax = plt.subplots(2,4, figsize = (10,5), sharex = True, sharey = True)
ax = np.ravel(ax)
# initialize j and k
j, k = 0, N
for i in range(M):
    # scatter the data
    ax[i].scatter(x_m[j:k],y_m[j:k])
    # set the x label
    ax[i].set_xlabel(f"x_{i}")
    # set the y label
    ax[i].set_ylabel(f"y_{i}",rotation = 0, labelpad = 15)
    # set the x axis limit
    ax[i].set_xlim(6,15)
    # set the y axis limit
    ax[i].set_ylim(7,17)
    # get the alpha of the group (mean of the posterior)
    alpha = trace_hm["alpha"][:,i].mean()
    # get the beta of the group (mean of the posterior)
    beta = trace_hm["beta"][:,i].mean()
    # get the xrange for which to plot the line
    x_range = np.linspace(x_m.min(), x_m.max(), 10)
    # plot the regression line
    ax[i].plot(x_range, alpha + beta * x_range, c='k',label=f'y = {alpha:.2f} + {beta:.2f} * x')
    # update j,k 
    j += N
    k += N
plt.tight_layout()
plt.show()
     
     
    
    