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

# ---------------- transform the data ------------------------ #

# save the idx's
vers_idx = iris[iris["species"] == "versicolor"].index
seto_idx = iris[iris["species"] == "setosa"].index
log.info("The versicolor idx is %s", vers_idx)
log.info("The setosa idx is %s", seto_idx)
# get the x's 
x_vers = iris.loc[vers_idx,"petal_width"].values
x_seto = iris.loc[seto_idx,"petal_width"].values
log.info("The versicolor is %s", x_vers)
log.info("The setosa is %s", x_seto)
# get x and y as usual 
# keep only setosa and versicolor
iris = iris[(iris["species"] == "setosa")|(iris["species"] == "versicolor")]
y_0 = pd.Categorical(iris["species"]).codes
x_0 = iris["petal_width"].values

# ----------------- specify a probabilistic model for the data ---------------- # 

with pm.Model() as gen_model:
    # priors for the mean/sd of the distribution of x's for each class
    mu = pm.Normal("mu", mu = 0, sd = 10, shape = 2)
    sigma = pm.HalfNormal("sigma", 10)
    # specify the likelihood of the data
    setosa = pm.Normal("setosa", mu = mu[0], sd = sigma, observed = x_seto)
    versicolor = pm.Normal("versicolor", mu = mu[1], sd = sigma, observed = x_vers)
    # get the decision boundary
    db = pm.Deterministic("db", (mu[0] + mu[1])/2)
    # inference step 
    trace = pm.sample(1000)
    
# ----------------- analyze the posterior ------------------------- # 
    
with gen_model: 
    log.info("the trace summary of the model is: %s", az.summary(trace))
    az.plot_trace(trace)
    
# ----------------- plot ------------------------------------ # 

plt.figure()
plt.axvline(trace["db"].mean(0), ymax = 1, color = "C1")
db_hpd = az.hpd(trace["db"])
plt.fill_betweenx([0,1],db_hpd[0],db_hpd[1], color = "C1", alpha = 0.5)
plt.plot(x_0,np.random.normal(y_0,0.02),".",color = "k")
plt.ylabel("theta",rotation = 0)
plt.xlabel("petal width")
plt.show()


    