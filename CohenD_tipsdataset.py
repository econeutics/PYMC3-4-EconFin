#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import logging
import pandas as pd
import numpy as np 
import scipy.stats as ss 
import arviz as az
import pymc3 as pm 
import seaborn as sns

# --------------------------------- import the data --------------------------------------------- # 

data = pd.read_csv('./data/tips.csv')
log.info("The tips data tail is as follows: %s", data.tail())
# plot the data by day
sns.violinplot(x = "day", y = "tip", data = data)

# --------------------------------- set variables -------------------------------------- #

# get the tips
tips = data.tip.values
# get the days and turn them into categories 0,1,2,3
days = pd.Categorical(data["day"],categories = ["Thur","Fri","Sat","Sun"]).codes
# get a variable equal to the number of categories 
cat_number = len(np.unique(days))

# ------------------------- specify the probabilistic model ------------------------ # 

with pm.Model() as model:
    # set the prior for the location parameter
    mu = pm.Normal("mu", mu = 0, sd = 10, shape = cat_number)
    # set the prior for the scale parameter
    sigma = pm.HalfNormal("sigma", sd = 10, shape = cat_number)
    # specify the likelihood of the data
    obs = pm.Normal("obs", mu = mu[days], sigma = sigma[days], observed = tips)
    # inference step 
    trace = pm.sample(1000)
    
# ------------------------- analyse the posterior ------------------------------- # 
    
with model: 
    # get the MAP estimates
    map_estimates = pm.find_MAP()
    log.info("The map estimates are: %s", map_estimates)
    # print the trace
    log.info("The summary of the mu trace with shape %s is: %s",trace["mu"].shape,trace["mu"])
    log.info("The summary of the sigma trace with shape %s is: %s",trace["sigma"].shape,trace["sigma"])
    # print a summary of the results
    log.info("The summary of the posterior is : %s", az.summary(trace))
    az.plot_trace(trace)
    
# ------------------- plot the difference between the posterior means and std ------------------------------- # 
    
with model: 
    # initialize a normal variable 
    dist = ss.norm()
    # initialize a plot with 3times2 figures
    _,ax = plt.subplots(3,2,figsize=(14,8), constrained_layout = True)
    # get the combinations of elements to be compared
    comparisons = [(i,j) for i in range(4) for j in range (i+1,4)]
    pos = [(k,l) for k in range(3) for l in (0,1)]
    # iterate over the elements to be compared and the graph positions 
    for (i,j),(k,l) in zip(comparisons,pos):
        # get the difference between the draws from the posterior 
        means_diff = trace["mu"][:,i] - trace["mu"][:,j]
        # get the D_cohen for each draw from the posterior and then get the mean
        d_cohen = (means_diff/np.sqrt((trace["sigma"][:,i]**2 + trace["sigma"][:,j]**2)/2)).mean()
        # get the probability of superiority 
        ps = dist.cdf(d_cohen/(2**0.5))
        az.plot_posterior(means_diff,ref_val = 0, ax = ax[k,l])
        # set the graph title
        ax[k,l].set_title(f"$\mu_{i} - \mu_{j}$")
        # get the legend specifics
        ax[k, l].plot(0, label=f"Cohen's d = {d_cohen:.2f}\nProb sup = {ps:.2f}",alpha=0)
        ax[k, l].legend()
        
        

