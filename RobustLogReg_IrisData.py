#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import logging 
import numpy as np 
import pandas as pd
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt 

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ---------------- import the iris data ----------------------------- # 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_width", data = iris, jitter = True)

# ---------------- transformations ---------------------------------- #

iris = iris.query("species == ('setosa','versicolor')")
y_0 = pd.Categorical(iris["species"]).codes
x_0 = iris["petal_length"].values
y_0 = np.concatenate((y_0,np.ones(6,dtype = int)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
# center the data
x_c = x_0 - x_0.mean()
plt.figure()
plt.plot(x_c,y_0,"o",color = "k")
plt.show()

# ---------------- specify a probabilistic model ---------------- # 

with pm.Model() as model_rlg:
    # set the priors for the parameters
    alpha = pm.Normal("alpha",mu = 0, sd = 10)
    beta = pm.Normal("beta",mu = 0, sd = 10)
    mu = alpha + beta * x_c
    # get the theta
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    # get the decision boundary 
    db = pm.Deterministic("db", - alpha/beta)
    # get a prior for the probability of an error
    pie = pm.Beta("pie",1,1)
    # finally get the probability of being of class 1
    p = pie * 0.5 + (1 - pie) * theta
    # specify the likelihood of the data
    y_obs = pm.Bernoulli("y_obs", p = p, observed = y_0)
    # inference step
    trace = pm.sample(1500)
    
# ------------------- analyse the posterior distribution -------------- # 

with model_rlg:
    log.info("The summary on the trace is as follows: %s", az.summary(trace, var_names = ["alpha","beta","db","pie"]))
    az.plot_trace(trace, var_names = ["alpha","beta","db","pie"])
    
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

    
    
    
    