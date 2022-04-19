#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:44:55 2020

@author: econeutics

"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss 
import pymc3 as pm 
import arviz as az

# --------------------------- define the 2nd Ascombes's quartet data ------------------------------------ # 

# define arrays of x and y values from the second Ascombe's quartet 
x = np.array([10,8,13,9,11,14,6,4,12,7,5])
y = np.array([9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74])
# center the x data
x = x - x.mean()
# plot the second Ascombe's quarter 
fig, ax = plt.subplots(figsize = (12,5))
ax.scatter(x,y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("The second Ascombe's quartet")
plt.tight_layout()
plt.show()
 
# ------------------------- specify a probabilistic model for the data -------------------------------- # 

with pm.Model() as polyn_model:
    # set the parameters prior
    alpha = pm.Normal("alpha", mu = y.mean(), sd = 1)
    beta_1 = pm.Normal("beta_1", mu = 0, sd = 1)
    beta_2 = pm.Normal("beta_2", mu = 0, sd = 1)
    sigma = pm.HalfCauchy("sigma",5)
    # model the mean of the data
    mu = alpha + beta_1 * x + beta_2 * x**2
    # specify the likelihood of the data
    y_obs = pm.Normal("y_obs",mu = mu, sigma = sigma, observed = y)
    # inference step
    trace = pm.sample(2000)
    
# ------------------------ plot the results --------------------------- # 
    
fig, ax = plt.subplots()
x_p = np.linspace(-6,6)
y_p = trace["alpha"].mean() + trace["beta_1"].mean() * x_p + trace["beta_2"].mean() * x_p**2
plt.scatter(x,y)
plt.plot(x_p,y_p, c = 'C1')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
    
    
    