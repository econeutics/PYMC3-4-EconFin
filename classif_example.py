import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import logging
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pymc3.gp.util import plot_gp_dist

# ----------------- initialize the logger -------------------------------------------- #

# initialize the logger
log = logging.getLogger(__name__)

# set the path where to import the data
data_name = "/Users/econeutics/Desktop/regression_data.xlsx"

# configure the logger
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ----------------- import data ------------------------------------------------------ #

data = pd.read_excel(data_name,index_col = 0, header = [0])

# ----------------- bayesian analysis ------------------------------------------------ #

# set a group variable
data["group"] = pd.Categorical(data["pan.substage"] + data["pan.geo"])
data["group_codes"] = pd.Categorical(data["pan.substage"] + data["pan.geo"]).codes
print(data[["group","group_codes"]].drop_duplicates())
# get the number of existing groups in the dataset
groups_number = len(np.unique(data["group"]))
# idx
idx = data["group_codes"].values
detrend = data["detrend"].values
other = data["other"].values
vintages = data["vintage"].values
# Get the independent variable
X = data.as_matrix(columns = ["detrend","other","vintage"])
# get the PMEs
Y = data["sw PME"].values
print(X.shape)
print(Y.shape)
print(idx.shape)
# --------------------------------- plot the data --------------------------------------------- #
#ax = plt.axes(projection = '3d')
ax = plt.axes()
x_line = X[:,0]
y_line = X[:,1]
z_line = Y
ax.scatter(x_line,z_line,c = idx)
plt.show()
# ---------------------------------------------------------------------------------------------- #
# build the bayesian model
with pm.Model() as model:
    alpha_mean = pm.Normal("alpha_mean",1,1)
    alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = 10)
    beta_fundraising = pm.Normal("beta_fundraising",0,10)
    beta_other = pm.Normal("beta_other",0,10,shape=groups_number)
    alpha = pm.Normal("alpha",alpha_mean,alpha_sigma,shape=groups_number)
    sigma = pm.HalfNormal("sigma", sigma = 1)
    ls = pm.HalfNormal("ls",sigma = 10)
    eta = pm.HalfNormal("eta",sigma = 10)
    #vu_ = pm.HalfNormal("vu_",sigma = 10)
    #vu = pm.Deterministic("vu",vu_ + 1)
    # initialize covariance of the Gaussian process
    cov_func = eta * pm.gp.cov.ExpQuad(input_dim = 3,ls = ls, active_dims= [2])
    # initialize the gp
    gp = pm.gp.Latent(cov_func = cov_func)
    # get the mv normal based on the sample
    f = gp.prior("f",X = X)
    # get the mu
    mu = pm.Deterministic("mu", alpha[idx] + beta_fundraising * X[:,0] + beta_other[idx] * X[:,1] + f)
    # likelihood
    y_obs = pm.Normal("y_obs",mu = mu, sigma = sigma, observed = Y)

with model:
    trace = pm.sample(cores = 1)
    sum = az.summary(trace,var_names=["alpha_mean","alpha_sigma","beta_fundraising","alpha","sigma","beta_other"])
    az.plot_trace(trace,var_names=["alpha_mean","alpha_sigma","beta_fundraising","sigma","beta_other"])
    az.plot_posterior(trace,var_names=["alpha_mean","alpha_sigma","beta_fundraising","sigma","beta_other"])
    plt.show()

# ---------------------------- study the shape of the trace ---------------------------------------------------- #

with model:
    for i in ["f", "eta", "ls", "sigma", "alpha", "beta_other", "beta_fundraising", "alpha_sigma", "alpha_mean","mu"]:
        log.info("The shape of trace for %s is: %s", i, trace[i].shape)

# ---------------------------- get the gp process prediction on the existent sample ---------------------------- #

with model:
    fig = plt.figure(figsize = (12,5)); ax = fig.gca()
    #print(X[:,2][:,None])
    #plot_gp_dist(ax,trace["f"],X[:,2][:,None])
    plt.plot(X[:,2],Y,'o',label = "True values")
    # check whether the dara looks normal after subtracting the estimated mean
    print(trace["mu"].mean(0))
    print(trace["mu"].mean(0).shape)
    print(trace["f"].mean(0).mean(0).shape)
    print(trace["f"].mean(0).mean(0))
    f_mean = trace["f"].mean(0).mean(0)
    noise = Y - trace["mu"].mean(0).mean(0)
    plt.plot(X[:,2],f_mean,'o',label = "f mean estimated")
    plt.plot(X[:,2],noise,'o',label = "noise")
    plt.legend()
    plt.show()

"""
with model:
    f_pred = gp.conditional("f_pred",Xnew=X)
    pred_samples = pm.sample_posterior_predictive(trace,vars [f_pred])
    print(pred_samples)
    print(pred_samples.shape)
    _,ax = plt.subplots(figsize = (10,6))
    az.plot_hpd(X[:,2],pred_samples["f_pred"],color = "C2")
    ax.scatter(X[:,2],Y)
    plt.show()
"""
# -------------------------- study the covariance of vintages ---------------------------------------------- #

# first plot the trace of ls and eta
az.plot_trace(trace,var_names=["ls","eta"])
# get the specific values of the trace
trace_ls = trace["ls"]
trace_eta = trace["eta"]
# get the xrange
xrange = np.linspace(0,X[:,2].max() - X[:,2].min())
# get the median covariance
_,ax = plt.subplots(1,1,figsize = (8,5))
ax.plot(xrange,np.median(trace_eta) * np.exp(-np.median(trace_ls) * xrange**2),lw = 3)
ax.plot(xrange, (trace_eta[::20][:,None] * np.exp(-trace_ls[::20][:,None] * xrange**2)).T,"C0",alpha=.1)
ax.set_xlabel("Vintage distance")
ax.set_ylabel("covariance")
plt.show()



# -----------------------
with model:
    ppc = pm.sample_posterior_predictive(trace,samples = 2000,model=model)
    data_ppc = az.from_pymc3(trace=trace,posterior_predictive=ppc)
    ax = az.plot_ppc(data_ppc,figsize = (12,6),mean=True)
    plt.show()

