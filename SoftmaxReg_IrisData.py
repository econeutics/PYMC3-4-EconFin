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
import theano as tt
from theano.tensor.nnet.nnet import softmax

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ---------------- import the iris data ----------------------------- # 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_width", data = iris, jitter = True)

# ---------------- transformations ----------------------- #

y_s = pd.Categorical(iris["species"]).codes
x_s = iris[["petal_length","petal_width"]].values
# standardize the x's
x_s = (x_s - x_s.mean(0))/x_s.std(0)
# get the groups number
groups_number = len(np.unique(iris["species"]))

# --------------- specify the probabilistic model ------------------------- # 

with pm.Model() as softmax_model:
    alpha = pm.Normal("alpha", mu = 0, sd = 10, shape = groups_number - 1)
    beta = pm.Normal("beta", mu = 0, sd = 10, shape = (x_s.shape[1],groups_number - 1))
    alpha_f = tt.tensor.concatenate([[0] ,alpha])
    beta_f = tt.tensor.concatenate([np.zeros((x_s.shape[1],1)) , beta], axis=1)
    # get the mu 
    mu = pm.Deterministic("mu", alpha_f + pm.math.dot(x_s,beta_f))
    # apply the softmax function to the mu 
    theta = softmax(mu)
    # specify the likelihood of the data
    y_obs = pm.Categorical("y_obs", p = theta, observed = y_s)
    # inference step 
    trace = pm.sample()
    
# -------------- check how many cases are classified correctly ----------- # 
    
data_pred = trace["mu"].mean(0)
log.info("The data pred is: %s", data_pred)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0) for point in data_pred]
print(f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}')
