#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------- first of all let us create some synthetic data ---------------------- #

# set true params
size = 200
true_alpha = 1
true_beta = 2
# generate data
x = np.linspace(0,1,size)
true_regression = true_alpha + true_beta * x 
y = np.random.normal(loc = true_regression, scale = 0.5
                     )
# plot the true data
_,ax = plt.subplots()
ax.plot(x,true_regression, color = "k", label = "true regression line")
ax.plot(x,y,".", color = "C1", label = "sampled data")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.legend()
plt.show()
# save the data in a dictionary 
data = dict(x = x, y = y)

# -------------- use the GLM module the estimate the model ------------------------ # 

with pm.Model() as model:
    # specify the regression model
    pm.glm.GLM.from_formula("y ~ x", data)
    # inference step 
    trace = pm.sample(3000)
    
# ------------- let's just analyse the posterior of the model --------------------- # 
    
with model:
    # let's just analyse the posterior of parameters
    az.plot_trace(trace)
    # plot posterior predictions
    plt.figure()
    plt.plot(x, y, 'x', label='data')
    pm.plot_posterior_predictive_glm(trace, samples = 100, label = "posterior predictive regression lines")
    plt.plot(x, true_regression, label='true regression line', lw=3., c='y')
    plt.title('Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
# ------------ create some new data with outliers ----------------------------- # 
    
# set true params
size = 200
true_alpha = 1
true_beta = 2
# generate data
x = np.linspace(0,1,size)
true_regression = true_alpha + true_beta * x 
y = np.random.normal(loc = true_regression, scale = 0.5)
# Add outliers
x_out = np.append(x, [.1, .15, .2])
y_out = np.append(y, [8, 6, 9])
# plot the true data
_,ax = plt.subplots()
ax.plot(x,true_regression, color = "k", label = "true regression line")
ax.plot(x_out,y_out,".", color = "C1", label = "sampled data")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.legend()
plt.show()
# save the data in a dictionary 
data = dict(x = x_out, y = y_out)

# -------------- use the GLM module the estimate a non-robust model ------------------------ # 

with pm.Model() as nr_model:
    # specify the regression model
    pm.glm.GLM.from_formula("y ~ x", data)
    # inference step 
    nr_trace = pm.sample(3000)

# -------------- use the GLM module the estimate a non-robust model ------------------------ # 

with pm.Model() as r_model:
    # set the t-distribution as the distribution family to be used for the likelihood
    family = pm.glm.families.StudentT()
    # specify the regression model
    pm.glm.GLM.from_formula("y ~ x", data, family = family)
    # inference step 
    r_trace = pm.sample(3000)

# -------------- analyse the posterior of both models ----------------------------- #
for model,trace in zip([nr_model,model],[nr_trace,r_trace]):
    with model:
        # let's just analyse the posterior of parameters
        az.plot_trace(trace)
        # plot posterior predictions
        plt.figure()
        plt.plot(x_out, y_out, 'x', label='data')
        pm.plot_posterior_predictive_glm(trace, samples = 100, label = "posterior predictive regression lines")
        plt.plot(x, true_regression, label='true regression line', lw=3., c='y')
        plt.title('Posterior predictive regression lines')
        plt.legend(loc=0)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()