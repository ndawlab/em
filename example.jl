# julia EM model fitting example, Nathaniel Daw 6/2026

####### TO RUN MULTITHREADED YOU MUST SET ENVIRONMENT VARIABLE JULIA_NUM_THREADS
####### BEFORE STARTING JULIA OR JUPYTER-NOTEBOOK

# eg in linux/bash:
#      export JULIA_NUM_THREADS=\`nproc\`; julia

# or just run julia with --threads=auto
###### setup 

full = false    # Maintain full covariance matrix (vs a diagional one) at the group level
emtol = 1e-3    # stopping condition (relative change) for EM

# to install the package run:
# import Pkg
# Pkg.add(url="https://github.com/ndawlab/em.git/")

# load the package
using EM

# this loads additional packages used in examples below
# if you don't have them installed, you can install them with Pkg.add("packagename")
using Statistics
using Random
using GLM
using DataFrames

###### Q learning example

# simulate some  qlearning data

Random.seed!(1234); # (for repeatability)

NS = 100;
NT = 200;
NP = 2;

params = zeros(NS,NP);

cov = randn(NS); # simulated between-subject variable, e.g. age or IQ
cov = cov .- mean(cov);

cov2 = randn(NS); # simulated between-subject variable, e.g. age or IQ
cov2 = cov2 .- mean(cov2);

# subject level parameters

params[:,1] = 1 .+ 0.5 * randn(NS) + cov; # softmax  temp: mean 1, effect of cov
params[:,2] = 0 .+ 1 * randn(NS) + cov2;  # learning rate: mean 0, effect of cov2

c = zeros(Int64,NS*NT);
r = zeros(Int64,NS*NT);
s = zeros(Int64,NS*NT);

for i = 1:NS
	(c[(i-1)*NT+1:i*NT],r[(i-1)*NT+1:i*NT]) = simq(params[i,:],NT);
	s[(i-1)*NT+1:i*NT] .= i;
end

data = DataFrame(sub=s,c=c,r=r);
subs = 1:NS;

# design matrix specifying the group level model
# this is replicated once for each model parameter
#
# in particular for each subject-level parameter x_ij  (subject i, parameter j)
#
# x_ij ~ Normal(X beta_j, Sigma)
#
# thus X has a row for each subject and a column for each predictor
# in the simplest case where the only predictor is an intercept, X = ones(NS)
# then beta_j specifies the group-level mean for parameter j
#
# but in this example we have two covariates that vary by subject
# so x_ij = beta_1j + beta_2j * cov_i + beta_3j * cov2_i
# and we infer the slopes beta for each parameter j as well as the intercept
#
# so we have a design matrix with 3 columns, and a row per subject:

X = [ones(NS) cov cov2];

# note: when you have no covariates (only intercepts) omit the brackets to get a column vector

# X = ones(NS)

# starting points for group level parameters
# betas: one column for each parameter, one row for each regressor (so here: 3 rows, 2 columns)
# make sure these are floats
# note: if you have a single predictor you need a row vector (length: # params)
# eg betas = [0. 0.];
# and if there is also only a single model parameter and no covariates, then betas is a scalar
# eg betas = 0.

startbetas = [1. 0; 0 0; 0 0]

# sigma: one element starting variance for each model parameter (this is really variance not SD)
# if there is only one model parameter it needs to be a length-one vector eg. sigma = [5.]

startsigma = [5., 1]

##### estimation and standard errors

# Define the EM Model (data, subjects, design matrix, nparam, and likelihood function)
model = EMModel(data, subs, X, 2, qlik; reg_names=["Intercept", "Cov1", "Cov2"], param_names=["Temp", "LR"])

# Fit the model
# (Returns an EMFit structure containing: betas, sigma, x, l, h, and the model)
fit = em(model; startbetas=startbetas, startsigma=startsigma, emtol=emtol, full=full)

# Standard errors on the subject-level means, based on an asymptotic Gaussian approx 
# (these may be inflated for small n)
# returns an EMErrors structure containing standard errors (ses), p-values, and covmtx.
errs = emerrors(fit)

# another way to get a p value for a covariate, by omitting it from the model and regressing
# this seems to work better when full=false
# in general not super well justified and can clearly be biased in some cases
# but works well in practice as long as you avoid the bias cases (which are pretty obvious)
X2 = ones(NS);
startbetas2 = [0. 0.];
startsigma2 = [5., 1];
model2 = EMModel(data, subs, X2, 2, qlik; reg_names=["Intercept"], param_names=["Temp", "LR"])
fit2 = em(model2; startbetas=startbetas2, startsigma=startsigma2, emtol=emtol, full=full);

display(lm(@formula(temp~cov+cov2),DataFrame(temp=fit2.x[:,1],cov=cov,cov2=cov2)))
display(lm(@formula(lr~cov+cov2),DataFrame(lr=fit2.x[:,2],cov=cov,cov2=cov2)))
# again the first covariate is significant for temp and the second for lr

## model selection/comparison/scoring

# Laplace approximation to the aggregate log marginal likelihood of the whole dataset
# marginalized over the individual params
lml(fit)

# to compare these between models you need to correct for the group level free parameters
# either aic or bic (this is Quentin Huys' IBIC or IAIC, i.e. the subject level
# params are marginalized by laplace approx, and aggregated, and the group level
# params are corrected by AIC or BIC)

ibic(fit, NS*NT)
iaic(fit)

# or by computing unbiased per subject marginal likelihoods via cross validation.
# you can do paired t tests on these between models
# these are also appropriate for SPM_BMS etc
liks = loocv(fit; emtol=emtol, full=full)
sum(liks)
# note that iaic does an excellent job of predicting the aggregate held out likelihood
# but importantly these are per subject scores that you can compare in paired tests
# across models as per Stephan et al. random effects model comparison

