#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import numpy as np 
import pandas as pd 
import scipy.stats as ss 
import matplotlib.pyplot as plt 
import arviz as az
import pymc3 as pm
import logging 

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------------- import the data ----------------------------------- # 

data = pd.read_csv('https://stats.idre.ucla.edu/stat/data/fish.csv')
# get the column of the data 
log.info("The columns in the data are as follows: %s", data.columns)
# get a description of the dataset
log.info("The description of the fishing data is as follows: %s", data.describe())
# plot the relationship between number of child/camper and counting
fig,ax = plt.subplots(1,2)
ax[0].plot(data["child"],data["count"],".")
ax[1].plot(data["camper"], data["count"], ".")
ax[0].set_xlabel("# children") ; ax[0].set_ylabel("# fishes")
ax[1].set_xlabel("campes yes/no") ; ax[1].set_ylabel("# fishes")
plt.subplots_adjust(wspace = 0.5)
ax[0].set_title("The relationship between # children and # fishes", fontsize = 8)
ax[1].set_title("The relationship between camper and # fishes",fontsize =8)
  
# --------------------------- transformations ----------------------------------- #

fishes = data["count"].values
children = data["child"].values
camper = data["camper"].values
  
# --------------------------- specify a probabilistic model ----------------------------------- # 

with pm.Model() as zip_regression:
    # get the priors on the parameters
    alpha = pm.Normal("alpha", mu = 0, sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10, shape = 2)
    # get the prior on the inflation coefficient
    psi = pm.Beta("psi", 1 , 1)
    # get the theta
    theta = pm.Deterministic("theta", pm.math.exp(alpha + beta[0] * children + beta[1] * camper))
    # specify the likelihood of the data
    y_obs = pm.ZeroInflatedPoisson("y_obs", psi, theta, observed = fishes)
    # inference step
    trace = pm.sample(1500)
    
# -------------------------- analyse the posterior --------------------------------------- # 
    
with zip_regression:
    log.info("The summary of the trace is as follows: %s", az.summary(trace,var_names = ["alpha","beta","psi"]))
    az.plot_trace(trace,var_names = ["alpha","beta","psi"])
    
# -------------------------- plot --------------------------------------- # 

plt.figure()
# initialize data to plot
children = [0,1,2,3,4]
fish_count_pred_0 = []
fish_count_pred_1 = []
for n in children:
    # get the prediction in case no camper 
    no_camper = trace["alpha"] + trace["beta"][:,0] * n
    # get the prediction in case of camper
    camper = no_camper + trace["beta"][:,1]
    # append predictions
    fish_count_pred_0.append(np.exp(no_camper))
    fish_count_pred_1.append(np.exp(camper))
plt.plot(children,fish_count_pred_0,'C0.',alpha = 0.01)
plt.plot(children,fish_count_pred_1,'C1.',alpha = 0.01)
plt.xticks(children)
plt.xlabel("Number of children")
plt.ylabel("fishes caught")
plt.plot([], 'C0o', label='without camper')
plt.plot([], 'C1o', label='with camper')
plt.legend()
plt.show()