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
import logging 

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ---------------------- import the data ----------------------------- 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_width", data = iris, jitter = True)

# ---------------------- transformations ------------------------ #

# keep only setosa and versicolor
iris = iris[(iris["species"] == "versicolor")|(iris["species"] == "virginica")]
# set the dependant variable
y_0 = pd.Categorical(iris["species"]).codes
# set the independent variable
x_0 = iris[["petal_length","petal_width"]].values + 5
# center the independent variable 
x_c = x_0 - x_0.mean(0)
#log.info("The centered data is: %s", x_c)

# --------------------- specify the probabilistic model --------------------- #

with pm.Model() as MultLog_model:
    # specify the priors on the parameters 
    alpha = pm.Normal("alpha", mu = 0,sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 2, shape = x_c.shape[1])
    # specify the value of theta 
    theta = pm.Deterministic("theta", pm.math.sigmoid(alpha + pm.math.dot(x_c,beta)))
    # specify a decision boundary for the data
    db = pm.Deterministic("db", - alpha/beta[1] - beta[0]/beta[1] * x_c[:,0])
    # specify the likelihood of the data
    y_obs = pm.Bernoulli("y_obs",p = theta, observed = y_0)
    # inference step 
    trace = pm.sample(2000,tune = 1500)
    
# ---------------------- analyse the posterior ---------------------------- # 
    
with MultLog_model:
    # analyse the summary 
    log.info("The summary of the trace is as follows: %s", az.summary(trace,var_names = ["alpha","beta"]))
    # plot the joint posterior
    az.plot_joint(trace, kind = "kde", var_names = ["beta"])
    
# ---------------------- plot the data with the decision boundary ------------- # 

# initialize a figure
plt.figure(figsize = (12,5))
# get the index to order the independent variable    
idx = np.argsort(x_c[:,0])
# get the mean of the decision boundary to plot
db = trace["db"].mean(0)[idx]
# scatter the true data
plt.scatter(x_c[:,0],x_c[:,1],c = [f'C{x}' for x in y_0])
# plot the decision boundary
plt.plot(x_c[:,0][idx], db, c = "k")
# get the hpd
az.plot_hpd(x_c[:,0],trace["db"], color = "k")
plt.show()



    
