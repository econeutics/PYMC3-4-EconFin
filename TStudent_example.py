#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import pandas as pd 
import numpy as np 
import scipy.stats as ss 
import logging
import pymc3 as pm 
import arviz as az 

#  ------------------------ analysis of a t-distribution - see how the curve depends on the degrees of freedom ---------------------------------#
# initialize a figure
plt.figure(figsize = (10,6))
# set the x values at which to evaluate the distributions
x_values = np.linspace(-10,10,500)
# loop over three combinations of degrees of freedom: 2,3,30
for df in [1,3,30]:
    # define the t distribution
    dist = ss.t(loc=0,scale=1,df=df)
    # calculate the probability density function at each point
    x_pdf = dist.pdf(x_values)
    # plot the probability density function 
    plt.plot(x_values,x_pdf,label = fr"$\nu = {df}$", lw = 3)
# define a normal rv with same location and scale
normal = ss.norm(loc=0,scale=1)
# plot the normal pdf 
plt.plot(x_values,normal.pdf(x_values),label = fr"$\nu = \infty$",lw = 3)
plt.legend()

# --------------------------- generate data from a t_distribution ------------------------------ # 

# set the true values of the population
loc_true = 0 ; scale_true = 1 ; df_true = 3 
# set the number of draws from the population 
draws = 1000
# draw from the population 
x_values = ss.t.rvs(size = draws, loc = loc_true, scale = scale_true,  df = df_true)
log.info("The draws from the population are: %s", x_values)
 
# --------------------------- specify a Gaussian model for the data ------------------------- # 

with pm.Model() as gaussian_model:
    # set the loc prior
    mu = pm.Uniform("mu",-100,100)
    # set the scale prior
    sigma = pm.HalfNormal("sigma",10)
    # specify the likelihood
    obs = pm.Normal("obs",mu = mu, sigma = sigma, observed = x_values)
    # inference step
    trace_gaussian = pm.sample(1000,tune = 1000)
    
# --------------------------- specify a T-student model for the data ----------------------- # 

with pm.Model() as t_model:
    # set the loc prior
    mu = pm.Uniform("mu",-100,100)
    # set the scale prior
    sigma = pm.HalfNormal("sigma",10)
    # set the df prior
    df = pm.Exponential("df",1/30)
    # specify the likelihood
    obs = pm.StudentT("obs",mu = mu, sigma = sigma, nu = df, observed = x_values)
    # inference step
    trace_t = pm.sample(1000,tune = 1000)
    
# ------------------------- analyse the posteriors ------------------------------------------- # 

for model, trace in zip([gaussian_model,t_model],[trace_gaussian,trace_t]):
    with model:
        # get the info on the trace
        log.info("The trace for the model is:%s,%s",trace["mu"],trace["sigma"])
        log.info("The trace shape for the model is:%s,%s",trace["mu"].shape,trace["sigma"].shape)
        # get the summary on the trace
        log.info("The summary on the trace is: %s",az.summary(trace))
        # analyze the trace KDE and the MCMC draws
        az.plot_trace(trace)
        
# ------------------------- compare the two models based on how well the data fits ---------------------- #

for model, trace in zip([gaussian_model,t_model],[trace_gaussian,trace_t]):
    with model:
        # get the data draws from the posterior
        x_new = pm.sample_posterior_predictive(trace,1000)
        print(x_new["obs"].shape)
        # plot the results
        data_ppc = az.from_pymc3(trace = trace,posterior_predictive = x_new)
        az.plot_ppc(data_ppc,figsize = (12,6),mean = False)
        ax[0].legend(fontsize=15)
        plt.xlim(-30,30)
    
    
    
 

