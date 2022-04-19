#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:10:51 2020

@author: davideferri
"""

import logging 
import numpy as np 
import pandas as pd 
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ---------------------- import Iris data ----------------------------------- # 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_width", data = iris, jitter = True)

# --------------------- transformations ---------------------------------- # 

# keep only setosa and virginica species 
iris = iris.query("species == ('setosa','versicolor')")
#exclude the first 45 observations 
iris = iris.iloc[35:,:].reset_index(drop=True)
# get the y variable
y_0 = pd.Categorical(iris["species"]).codes
# get the x variable
x_0 = iris[["petal_length","petal_width"]].values + 5

# --------------------- specify a probabilistic model ---------------------- # 

with pm.Model() as log_model:
    # specify the prior 
    alpha = pm.Normal("alpha", mu = 0, sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10, shape = x_0.shape[1])
    # get the theta and db
    theta = pm.Deterministic("theta", pm.math.sigmoid(alpha + pm.math.dot(x_0,beta)))
    db = pm.Deterministic("db", -alpha/beta[1] - beta[0]/beta[1] * x_0[:,0])    
    # specify a likelihood
    y_obs = pm.Bernoulli("y_obs", p = theta, observed = y_0)
    # inference step 
    trace = pm.sample(1500,tune = 1000)
    
# --------------------- analyse the posterior -------------------------- # 

with log_model:
    log.info("the summary of the trace is: %s", az.summary(trace, var_names = ["alpha","beta"]))

# ---------------------- plot the data with the decision boundary ------------- # 

# initialize a figure
plt.figure(figsize = (12,5))
# get the index to order the independent variable    
idx = np.argsort(x_0[:,0])
# get the mean of the decision boundary to plot
db = trace["db"].mean(0)[idx]
# scatter the true data
plt.scatter(x_0[:,0],x_0[:,1],c = [f'C{x}' for x in y_0])
# plot the decision boundary
plt.plot(x_0[:,0][idx], db, c = "k")
# get the hpd
az.plot_hpd(x_0[:,0],trace["db"], color = "k")
plt.show()