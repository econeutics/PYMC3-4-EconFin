#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: econeutics
"""

import numpy as np 
import pandas as pd 
import scipy.stats as ss 
import logging 
import pymc3 as pm
import arviz as az

# ------------------ generate the synthetic data --------------------------- # 

# specify the number of observations in each of the three groups 
N_samples = [30,30,30]
# specify the number of observations equal to 1 in each of the three groups 
# we provide three different scenarios and repeat the analysis for each one
G_samples_1 = [18,18,18]
G_samples_2 = [3,3,3]
G_samples_3 = [18,3,3]
for G_samples in [G_samples_1,G_samples_2,G_samples_3]:
    # generate group indexes
    group_idx = np.repeat(np.arange(len(N_samples)),N_samples)
    log.info("the groups idx are: %s", group_idx)
    # generate the data for each group 
    data = []
    for i in range(len(N_samples)):
        data.extend(np.repeat([1,0],[G_samples[i],N_samples[i] - G_samples[i]]))
        log.info("The generated synthetic data is: %s", data)

# ------------------ specify a probabilistic model ---------------------- # 

    with pm.Model() as hierarchical_model: 
    # set the hyperpriors - location 
        mu = pm.Beta("mu",1,1)
        # set the hyperpriors - precision 
        k = pm.HalfNormal("k",10)
        # set the deterministic variables
        alpha = pm.Deterministic("alpha", mu * k)
        beta = pm.Deterministic("beta",(1 - mu) * k)
        # set the priors
        theta = pm.Beta("theta", alpha = alpha, beta = beta, shape = len(N_samples))
        # specify the likelihood of the observations 
        obs = pm.Bernoulli("obs",p = theta[group_idx], observed = data)
        # inference step 
        trace = pm.sample(2000)
    
# ------------------- analyse posterior --------------------------- à 

    with hierarchical_model:
        log.info("The trace summary is: %s", az.summary(trace))
        for var in ["theta","mu","k"]:
            log.info("the trace for %s with shape %s is: %s", var, trace[var].shape, trace[var])
        az.plot_trace(trace)
        az.plot_joint(trace,var_names = ["mu","k"], kind = "kde", fill_last = False)

    
    