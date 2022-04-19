#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



@author: econeutics
"""

import pandas as pd 
import scipy.stats as ss 
import numpy as np 
import pymc3 as pm 
import arviz as az
import logging 

# ------------------------- generate the data from the population ---------------------------------- # 

# configure logger
logging.basicConfig(filename='example.log',level=logging.DEBUG)
# set a random seed
np.random.seed(123)
# set the number of draws from the population
trials = 100
# set the true value of the parameter
theta_real = 0.35
# get the random draws from the population 
data = ss.bernoulli.rvs(p=theta_real,size=trials)
log.info("The random draws from the populations is: %s", data)

# ------------------------- specify the probabilistic model -------------------------------------- # 

with pm.Model() as bernoulli_model: 
    # set a Beta prior on the Bernoulli parameter
    theta = pm.Beta("theta",alpha = 1, beta = 1)
    # set a Bernoulli likelihood for the data
    y = pm.Bernoulli("obs", p = theta, observed = data)
    # inference step 
    trace = pm.sample(1000,random_seed = 123)
    
# ------------------------- analyze the posterior ----------------------------------------------- # 
    
with bernoulli_model:
    log.info("The trace is as follows: %s", trace["theta"])
    log.info("the dimensions of the trace is: %s", trace["theta"].shape)
    # see the summary of the trace 
    log.info("The summary of the trace is: %s", az.summary(trace))
    # plot a visual representation of the trace
    az.plot_trace(trace)
    # plot the posterior with the rope
    az.plot_posterior(trace,credible_interval = 0.9,rope = [0.45,0.55])
    # plot the posterior with a reference value 
    az.plot_posterior(trace,credible_interval = 0.9, ref_val = 0.5)
    
# ------------------------ loss function analysis -------------------------------------------- #
    
with bernoulli_model:
    # define a grid of points over which to evaluate the loss functions 
    grid = np.linspace(0,1,200)
    # define loss function a 
    loss_func_a = [np.mean(abs(i - trace["theta"])) for i in grid]
    # define loss function b 
    loss_func_b = [np.mean((i - trace["theta"])**2) for i in grid]
    # define asymmetric loss function, loss function c
    loss_func_c = []
    for i in grid:
        if i < 0.5:
            value = np.mean(np.pi * trace["theta"]/(i - trace["theta"]))
        else:
            value = np.mean(1/(i - trace["theta"]))
        loss_func_c.append(value)
    # plot the loss functions 
    for lossf,c in zip([loss_func_a,loss_func_b,loss_func_c],["C0","C1","C2"]):
        # get the argminimum loss value
        mini = np.argmin(lossf)
        # plot and highlight the minimum 
        plt.figure(figsize=(10,5))
        plt.plot(grid,lossf,c)
        plt.plot(grid[mini],lossf[mini],"o",color = c)
        plt.annotate("{:.2f}".format(grid[mini]),(grid[mini], lossf[mini] + 0.02), color = c)
        plt.yticks([])
        plt.xlabel("theta")
        plt.show()
        
    
                      

                      
                     