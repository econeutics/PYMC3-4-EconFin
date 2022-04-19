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
from theano import shared

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------  simulate data ----------------------------------------- # 

# set the random seed 
np.random.seed(123)
# set the number of draws 
N = 1000
# define the independent variable 
x = np.linspace(-50,50,N)
# define the true coefficients
alpha_true = 1 ; beta_true = 2 ; gamma_true = 1 ; delta_true = 0.5
# define the dependant variable
y = np.random.normal(loc = alpha_true + beta_true * x, scale = gamma_true + delta_true * np.abs(x))
# plot
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()


# -------------------- specify a probabilistic model for the data -------------- # 

with pm.Model() as vv_model:
    # specify the priors for the parameters
    alpha = pm.Normal("alpha", mu = 0, sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10)
    gamma = pm.HalfNormal("gamma", 10)
    delta = pm.HalfNormal("delta", 10)
    # set the independent variable as a shared variable
    x_shared = shared(x)
    # specify the mean of the observations
    mu = pm.Deterministic("mu", alpha + beta * x_shared)
    # specify the variance of the observations
    eps = pm.Deterministic("eps", gamma + delta * np.abs(x_shared))
    # set the likelihood of observations
    y_obs = pm.Normal("y_obs", mu = mu, sd = eps, observed = y)
    # inference step 
    trace = pm.sample(1000, tune = 2000)
    
# ------------------ study the posterior distribution of the parameters --------- # 
with vv_model:
    print(az.summary(trace, var_names = ["alpha","beta","gamma","delta"]))
    
# ----------------- plot analysis ---------------------------------------------- # 

fig, ax = plt.subplots(figsize = (12,5))
# scatter the true data
ax.scatter(x,y,alpha=0.3)
# get the mean of the posterior mean 
mu_m = trace["mu"].mean(0)
# plot the mean of the data 
ax.plot(x,mu_m,c = "k")
# get the mean standard deviation of the data
eps_m = trace["eps"].mean(0)
# color the area inside 1 std from the mean
ax.fill_between(x, mu_m + eps_m , mu_m - eps_m, color = "C1", alpha = 0.6)
# color the area inside 2 std from the mean 
ax.fill_between(x, mu_m + 2 * eps_m, mu_m - 2 * eps_m, color = "C1", alpha = 0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# --------------- predict and analyse --------------------------------- #

# predict values for new values of the predictors 
x_shared.set_value([20])
# sample from the posterior of y
ppc = pm.sample_posterior_predictive(trace,2000, model = vv_model)
# get the values
y_ppc = ppc["y_obs"][:,0]
# plot
fig, ax = plt.subplots(figsize = (12,5)) 
az.plot_kde(y_ppc)
ax.set_xlabel("y", fontsize = 25)
ax.set_ylabel("pdf",fontsize = 25)
plt.show()