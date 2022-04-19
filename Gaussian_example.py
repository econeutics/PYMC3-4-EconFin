#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: econeutics
"""

import pandas as pd 
import numpy as np 
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import logging

# ---------------------- generate drows from population ---------------------------- #

# set the random seed 
np.random.seed(123)
# set the true value of the parameters
mue_true = 1 ; sigma_true = 50
# set the number of draws 
draws = 1000
# get the draws from the population 
data = ss.norm.rvs(loc= mue_true, scale= sigma_true, size = draws)
log.info("the data is drawn is as follows: %s", data)
# plot the Kernel density estimation of the data
az.plot_kde(data)

# ------------------------ specify the probabilistic model --------------------------- #  

with pm.Model() as gaussian_model:
    # set the priors for the parameters
    mu = pm.Uniform("mu",-10,10)
    sigma = pm.HalfNormal("sigma",10)
    # get the likelihood
    y = pm.Normal("obs", mu = mu, sigma = sigma, observed = data)
    # inference step 
    trace = pm.sample(1000)
    
# ----------------------- analyse the posterior --------------------------------------- #
    
with gaussian_model:
    # show the trace
    log.info("The trace of mu is %s:", trace["mu"]), log.info("the shape is %s", trace["mu"].shape)
    log.info("The trace of sigma is %s:", trace["sigma"]), log.info("the shape is %s", trace["sigma"].shape)
    # show the trace summary
    az.summary(trace)
    # plot the trace KDE and MCMC draws
    az.plot_trace(trace)
    # plot the trace joint KDE 
    az.plot_joint(trace, kind="kde", fill_last = False)
    
# ------------------------ get samples of the data from the posterior ----------------- # 
    
with gaussian_model:
    # get the samples of the data
    y_new = pm.sample_posterior_predictive(trace)
    log.info("The samples from the data is: %s", y_new["obs"])
    log.info("The shape of the samples is: %s", y_new["obs"].shape)
    # visual check for whether the original sample makes sense given the posterior
    data_ppc = az.from_pymc3(trace=trace,posterior_predictive=y_new)
    ax = az.plot_ppc(data_ppc,figsize=(12,6),mean=False)
    ax[0].legend(fontsize=15)
    