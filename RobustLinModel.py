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

# ----------------------- get the data from the third Ascombe quartet -------------------- # 

# define variables
x = np.array([10,8,13,9,11,14,6,4,12,7,5])
y = np.array([7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73])
log.info("The third Ascombe's quartet is: %s", np.stack((x,y)).T)
# plot
fig,ax = plt.subplots()
ax.scatter(x,y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("The third Ascombe's quartet")
plt.show()
# center the x data
x = x - x.mean()

# ----------------------- specify a probabilistic model for the data ----------------------- # 

with pm.Model() as model_t:
    # set the prior over the intercept and the coefficient 
    alpha = pm.Normal("alpha", mu = y.mean(), sd = 1)
    beta = pm.Normal("beta", mu = 0, sd = 1)
    # set the prior over the errors variance
    sigma = pm.HalfNormal("sigma", 5)
    # set the prior on the degrees of freedom 
    vu_ = pm.Exponential("vu_", 1/29)
    vu = pm.Deterministic("vu", vu_ + 1)
    # get the likelihood on the data
    obs = pm.StudentT("obs", mu = alpha + beta * x, sigma = sigma, nu = vu, observed = y)
    # inference step 
    trace = pm.sample(2000)
    
# _------------------ compare the result of a simple linear regression (which assumes gaussian errors) and the robust linear regression ------------------ # 

# get the coefficient and intercept from a scipy linear regression
beta_c , alpha_c = ss.linregress(x,y)[:2]
# plot the non robust linear regression 
plt.plot(x, alpha_c + beta_c * x, c = "k", label = "Non-robust regression", alpha = 0.5)
# plot the data
plt.plot(x,y,'C0o')
# get the mean of the intercept from the posterior
alpha_m = trace["alpha"].mean()
# get the mean of the coefficient from the posterior
beta_m = trace["beta"].mean()
# plot the robust linear regression 
plt.plot(x, alpha_m + beta_m * x, c = "k" , label = "Robust linear regression")
# plot the variety of predicted results
az.plot_hpd(x,ppc["obs"])
# set the last details of the graph 
plt.xlabel("x")
plt.ylabel("y",rotation = 0)
#plt.legend(loc=2)
plt.tight_layout()
plt.show()

# ----------------- analyse the posterior -------------------- # 

with model_t: 
    az.plot_trace(trace, var_names = ["alpha","beta","sigma","vu"])
    # get the summary
    log.info("the trace summary is: %s", az.summary(trace))
    # let's also run a posterior predictive check 
    ppc = pm.sample_posterior_predictive(trace, samples = 2000)
    data_ppc = az.from_pymc3(trace = trace, posterior_predictive = ppc)
    ax = az.plot_ppc(data_ppc,figsize = (12,6), mean = True)
    plt.xlim(0,12)
    