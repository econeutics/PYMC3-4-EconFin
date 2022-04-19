#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: econeutics
"""

import pandas as pd
import numpy as np
import logging
import scipy.stats as ss
import pymc3 as pm
import arviz as az 
import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')
# ---------------------- import the data ----------------------------- 

data = pd.read_csv('./data/IndividualDetails_IndiaCOVID.csv')
log.info("The head of the COVID dataset is: %s", data.head())
log.info("The data columns are: %s", data.columns)
# drop all the observations with missing state or current status
data.dropna(subset=("detected_state","current_status"),inplace = True)
log.info("The current status values are: %s", np.unique(data["current_status"].astype(str)))
# keet only data of recovered/deceased
data = data[data["current_status"].isin(['Deceased','Recovered'])]
log.info("The dataset is: %s", data)
# get the states names 
states = np.unique(data["detected_state"])
log.info("The states are: %s", states)
# get the y variable
y = np.array((data["current_status"] == 'Deceased').astype(int))
# get the group idx
idx = pd.Categorical(data["detected_state"]).codes

# ---------------------- specify the probabilistic model / nonhierarchical ----------------- # 

with pm.Model() as nh_model:
    # set the prior
    theta = pm.Beta("theta", alpha = 1, beta = 1, shape = len(states))
    # set the likelihood 
    obs = pm.Bernoulli("obs",p = theta[idx], observed = y)
    # inference step 
    nh_trace = pm.sample(1000)
    
# ---------------------- specify the probabilistic model/ hierarchical --------------------- #

with pm.Model() as h_model:
    # set the hyperprior
    mu = pm.Beta("mu", 1,1)
    k = pm.HalfNormal("k",10)
    # set the prior
    theta = pm.Beta("theta", alpha = k * mu, beta = (1 - mu) * k, shape = len(states))
    # set the likelihood 
    obs = pm.Bernoulli("obs", p = theta[idx], observed = y)
    # inference step 
    h_trace = pm.sample(1000)
    
#------------------------ posterior analysis --------------------------------- # 
    
for model,trace in zip([nh_model,h_model],[nh_trace,h_trace]):
    with model:
        az.plot_trace(trace)
        
# --------------------- compare the theta posteriors between hierarchical and non hierarchical model ------------ #
        
axes = az.plot_forest([nh_trace,h_trace],model_names = ["nh","h"],var_names="theta",combined=False,colors="cycle")
y_lims = axes[0].get_ylim()
axes[0].vlines(h_trace["mu"].mean(),*y_lims) 
plt.show()   
    
