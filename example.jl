# julia EM model fitting, Nathaniel Daw 8/2019


###### setup 

parallel = true # Run on multiple CPUs. If you are having trouble, set parallel = false: easier to debug
full = false    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM

using Distributed
if (parallel)
	# only run this once
	addprocs()
end


# this loads the packages needed -- the @everywhere makes sure they 
# available on all CPUs 

@everywhere using DataFrames
@everywhere using SharedArrays
@everywhere using ForwardDiff
@everywhere using Optim
@everywhere using LinearAlgebra       # for tr, diagonal
@everywhere using StatsFuns           # logsumexp
@everywhere using SpecialFunctions    # for erf
@everywhere using Statistics          # for mean
@everywhere using Distributions
@everywhere using GLM

# change this to where you keep the code
@everywhere directory = "/users/ndaw/Dropbox (Princeton)/expts/julia em/git/em"

@everywhere include("$directory/emnew.jl");
@everywhere include("$directory/commonnew.jl");
@everywhere include("$directory/likfuns.jl")

###### Q learning example

# simulate some  qlearning data

using Random
Random.seed!(1234); # (for repeatability)

NS = 100;
NT = 200;
NP = 2;

params = zeros(NS,NP);

cov = randn(NS); # simulated between-subeject variable, e.g. age or IQ
cov = cov .- mean(cov);

cov2 = randn(NS); # simulated between-subeject variable, e.g. age or IQ
cov2 = cov2 .- mean(cov2);


# subject level parameters
params[:,1] = 1 .+ 0.5 * randn(NS) + cov;
params[:,2] = 0 .+ 1 * randn(NS) + cov2;

c = zeros(Int64,NS*NT);
r = zeros(Int64,NS*NT);
s = zeros(Int64,NS*NT);

for i = 1:NS
	(c[(i-1)*NT+1:i*NT],r[(i-1)*NT+1:i*NT]) = simq(params[i,:],NT);
	s[(i-1)*NT+1:i*NT] .= i;
end

data = DataFrame(sub=s,c=c,r=r);
subs = 1:NS;

# group level design matrix
# The new rule is that there is a single design matrix for all parameters
# one column per predictor
# here we have a mean (ones) and two covariates)

X = [ones(NS) cov cov2];

# note: for a single predictor omit the brackets to get a column vector

# X = ones(NS)

# starting points for group level parameters
# betas: one column for each parameter, one row for each regressor

betas = [0. 0; 0 0; 0 0]

# note: for a single predictor you need a row vector
# betas = [0. 0.];

# sigma: one element for each parameter (this is really variance not SD)
sigma = [5., 1]


##### estimation and standard errors

# fit the model
# (this takes: a data frame, a list of subjects, a group level design matrix, 
#  starting group level betas, starting group-level variance or covariance, a likelihood function
#  and some optional options)
#
# (return values: betas are the group level means
#  sigma is the group level *variance* or covariance
#  x is a matrix of per-subject parameters
#  l is the per-subject negative log likelihoods 
#  h is the *inverse* per subject hessians) 

(betas,sigma,x,l,h) = em(data,subs,X,betas,sigma,qlik; emtol=emtol, parallel=parallel, full=full);

# standard errors on the subject-level means, based on an asymptotic Gaussian approx 
# (these may be inflated for small n)

(standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,qlik)

# another way to get a p value for a covariate, by omitting it from the model and regressing
# this seems to work better when full=false
# in general not super well motivated

X2 = ones(NS);
betas2 = [0. 0.];
sigma2 = [5., 1];
(betas2,sigma2,x2,l2,h2) = em(data,subs,X2,betas2,sigma2,qlik; emtol=1e-5, parallel=parallel, full=full);

lm(@formula(beta~cov+cov2),DataFrame(beta=x2[:,1],cov=cov,cov2=cov2))
lm(@formula(beta~cov+cov2),DataFrame(beta=x2[:,2],cov=cov,cov2=cov2))

## model selection/comparison/scoring

# laplace approximation to the aggregate log marginal likelihood of the whole dataset
# marginalized over the individual params

ll1 = lml(x,l,h)

# to compare these between models you need to correct for the group level free parameters, either aic or bic

ibic(x,l,h,betas,sigma,NS*NT)
iaic(x,l,h,betas,sigma)

# or by computing unbiased per subject marginal likelihoods via cross validation.
# you can do paired t tests on these between models
# these are also appropriate for SPM_BMS etc

liks = loocv(data,subs,x,X,betas,sigma,qlik;emtol=emtol, parallel=parallel, full=full)
sum(liks)
