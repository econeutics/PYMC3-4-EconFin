#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: econeutics
"""

import numpy as np 
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import pymc3 as pm 
import arviz as az

# ------------------- plot some poisson distributions ---------------- # 

lambda_params = [0.5,1.5,3,8]
x = np.arange(0,max(lambda_params) * 3)
# iterate over the lambda parameters
for param in lambda_params:
    y = ss.poisson(param).pmf(x)
    plt.plot(x,y,"o-",label = f"lambda = {param:3.1f}")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# -------------------- generate synthetic data --------------------- # 

# set the number of draws
n = 1000
# set the true theta 
theta_real = 2.5
# set the zero-inflating factor 
psi_true = 0.5
# generate data from the ZIP model
counts = np.array([(np.random.random() > 1 - psi_true) * np.random.poisson(theta_real) for i in range(n)])

# --------------------- probabilistic method ---------------------- # 

with pm.Model() as zip_model:
    # specify the priors of the zero-inflated Poisson model
    psi = pm.Beta("psi", 1,1)
    theta = pm.Gamma("theta",2,0.1)
    # specify the likelihood of the data
    y = pm.ZeroInflatedPoisson("y",psi,theta,observed = counts)
    # inference step 
    trace = pm.sample(1500)

# ---------------------- analyse the posterior -------------- # 
    
with zip_model:
    az.plot_trace(trace)
    

