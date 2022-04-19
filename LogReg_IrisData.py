#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:51:18 2020

@author: davideferri
"""

import numpy as np 
import pandas as pd 
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import logging
import matplotlib.pyplot as plt 
import seaborn as sns

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ---------------------- import the data ----------------------------- 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_length", data = iris, jitter = True)

# ---------------------- transformations ------------------------ #

# keep only setosa and versicolor
iris = iris[(iris["species"] == "setosa")|(iris["species"] == "versicolor")]
# set the dependant variable
y_0 = pd.Categorical(iris["species"]).codes
# set the independent variable
x_0 = iris["petal_length"].values
# center the independent variable 
x_c = x_0 - x_0.mean()

# --------------------- specify the probabilistic model --------- # 

with pm.Model() as log_model:
    # set the priors 
    alpha = pm.Normal("alpha",mu = 0,sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10)
    # set the bernoulli parameter
    theta = pm.Deterministic("theta",pm.math.sigmoid(alpha + beta * x_c))
    # set the decision boundary 
    db = pm.Deterministic("db", -alpha/beta)
    # set the likelihood 
    y_obs = pm.Bernoulli("y_obs",p = theta, observed = y_0)
    # inference step
    trace = pm.sample(1000)

# -------------------- study the posterior distribution -------------------- # 

with log_model: 
    log.info("The summary on the trace is: %s", az.summary(trace, var_names = ["alpha","beta","db"]))
    az.plot_trace(trace, var_names = ["alpha","beta","db"])
    az.plot_joint(trace, kind = "kde", var_names = ["alpha","beta"])
    
    
# ------------------- plots ------------------------------------------- # 
    
# get the mean of theta
theta_mean = trace["theta"].mean(0)
# get idx
idx = np.argsort(x_c)
# plot the predicted p of the data
plt.figure()
plt.plot(x_c[idx],theta_mean[idx],color = "C2", lw = 3)
# set a vertical line at the mean of the decision boundary 
plt.vlines(trace["db"].mean(),0,1,color = "k")
# get the hpd of db
db_hpd = az.hpd(trace["db"])
plt.fill_betweenx([0,1],db_hpd[0],db_hpd[1], color = "k", alpha = 0.5)
plt.scatter(x_c,np.random.normal(y_0,0.02), marker='.', color=[f'C{x}' for x in y_0])
az.plot_hpd(x_c,trace["theta"],color = "C2")
plt.xlabel("petal length")
plt.ylabel("theta",rotation=0)
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))
plt.show()

