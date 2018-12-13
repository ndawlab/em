# julia EM model fitting, Nathaniel Daw 9/2017


###### setup

parallel = true # Run on multiple CPUs. If you are having trouble, set parallel = false: easier to debug
full = false    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM

if (parallel)
	using Distributed # this is now required since addprocs was moved into this package
	# only run this once
	addprocs()
end


using DataFrames
using ForwardDiff
using PyCall
using Distributions
using GLM
using SharedArrays
using LinearAlgebra
using SpecialFunctions
using StatsBase

# this loads the packages needed -- the @everywhere makes sure they
# available on all CPUs
@everywhere using DataFrames
@everywhere using ForwardDiff
@everywhere using PyCall
@everywhere using Distributions
@everywhere using GLM
@everywhere using SharedArrays
@everywhere using LinearAlgebra
@everywhere using SpecialFunctions
@everywhere using StatsBase
@everywhere PyCall.@pyimport scipy.optimize as so


# change this to where you keep the code
@everywhere directory = "/Users/yoelsanchezaraujo/Documents/em/julia_v1";

@everywhere include("$directory/em.jl");
@everywhere include("$directory/common.jl");
@everywhere include("$directory/likfuns.jl")

###### Q learning example

# simulate some  qlearning data
NS = 20;
NT = 200;
NP = 2;

gparams = zeros(NS,NP);

covm = randn(NS); # simulated between-subeject variable, e.g. age or IQ
covm = covm .- mean(covm);
# subject level parameters
gparams[:,1] = 1 .+ 0.5 * covm .+ 0.5 * randn(NS);
gparams[:,2] = randn(NS);

c = zeros(Int64,NS*NT);
r = zeros(Int64,NS*NT);
s = zeros(Int64,NS*NT);

for i = 1:NS
	(c[(i-1)*NT+1:i*NT],r[(i-1)*NT+1:i*NT]) = simq(gparams[i,:],NT);
	s[(i-1)*NT+1:i*NT] .= i;
end

data = DataFrame(sub=s,c=c,r=r);
subs = 1:NS;

# group level design matrix
# you specify this as a cell array ("Any[ ]") of design matrices, one for each subject level parameter
# these design matrices need not be the same.
# Here we have mean for both parameters ("ones[NS]") but include a covariate ("cov")
# for the first parameter only.
# If you have no covariates you can simplify this to designmatrix([ones(NS), ones(NS)])
# more covariates would look like [ones(NS), age, iq, weight]

X = designmatrix(Any[[ones(NS) covm], ones(NS)]);

# starting points for group level parameters
# here I also start with a cell array, one vector for each param
# (just to make counting easier) but the functions all actually just work with a single flattened list

betas = flatten(Any[[0.,0.], 0.]);  # == [0.,0.,0.]
sigma = [5.,1]; # these are variances, one for each subject level parameter. could also specify a cov mtx

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

(betas,sigma,x,l,h) = em(data,subs,X,Float64.(zeros(3)),sigma,qlik; emtol=emtol, parallel=parallel, full=full, quiet=true);

# standard errors on the subject-level means, based on an asymptotic Gaussian approx
# (these may be inflated for small n)

(standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,qlik)

# a more reliable but slow way to get standard errors using a bootstrap

(standarderrorsbs,pvaluesbs,pvaluesbsnp,covmtxbs) = emerrorsbs(
    data,subs,x,X,h,betas,sigma,qlik;
    parallel=parallel, full=full, nsamples=100, emtol=1e-3
)

# another way to get a p value for a covariate, by omitting it from the model and regressing

X2 = designmatrix([ones(NS), ones(NS)]);
betas2 = [0.,0.];
sigma2 = [5.,1];
(betas2,sigma2,x2,l2,h2) = em(data,subs,X2,betas2,sigma2,qlik; emtol=1e-5, parallel=parallel, full=full, quiet=true);

lm(@formula(beta~covm),DataFrame(beta=x2[1,:],covm=covm))

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

#### another example using a two-step task

# simulate some two step data

NS = 25;
NT = 200;
NP = 6;

params = zeros(NS,6);
cov = randn(NS);
cov = cov - mean(cov);

params[:,1] = 3/4 + 0.4 * cov + 0.5 * randn(NS);
params[:,2] = 3/4 + 0.5 * randn(NS);
params[:,3] = 3/4 + 0.5 * randn(NS);
params[:,4] = 3/4 + 0.5 * randn(NS);
params[:,5] =  0 + 0.5 * randn(NS);
params[:,6] = .5 + 0.5 * randn(NS);

c1 = zeros(Int64,NS*NT);
c2 = zeros(Int64,NS*NT);
s = zeros(Int64,NS*NT);
r = zeros(Int64,NS*NT);
sub = zeros(Int64,NS*NT);

for i = 1:NS
	(c1[(i-1)*NT+1:i*NT],s[(i-1)*NT+1:i*NT],c2[(i-1)*NT+1:i*NT],r[(i-1)*NT+1:i*NT]) = simseq(params[i,:],NT);
	sub[(i-1)*NT+1:i*NT] = i;
end

data = DataFrame(sub=sub,ch1=c1,ch2=c2,st=s,mn=r);
subs = 1:NS;

# fit it

# group level design matrix
# you specify this as a cell array ("Any[ ]") of design matrices, one for each subject level parameter
# these design matrices need not be the same.
# Here we have mean for both parameters ("ones[NS]") but include a covariate ("cov")
# for the first parameter only.
# If you have no covariates you can simplify this to designmatrix([ones(NS), ones(NS)])
# more covariates would look like [ones(NS), age, iq, weight]

X = designmatrix(Any[[ones(NS) cov], ones(NS), ones(NS), ones(NS), ones(NS), ones(NS)]);

# starting points for group level parameters
# I also start with a cell array, one vector for each param
# (just to make counting easier) but the functions take it flattened into a single vector

betas = flatten(Any[[1.,0], 1., 1., 1., 1., 1. ]);
sigma = [5.,5,5,5,1.0,5];


##### estimation and standard errors

# fit the model
# (this takes: a data frame, a list of subjects, a group level design matrix,
#  starting group level betas, starting group-level variance or covariance, a likelihood function
#  and some optional options)
#
# (return values: betas are the group level means
#  sigma is the group level *variance* or covariance
#  x is a matrix of per-subject parameters
#  l is the per-subject NLLS.
#  h is the *inverse* per subject hessians)

(betas,sigma,x,l,h) = em(data,subs,X,betas,sigma,seqlik; emtol=emtol,parallel=parallel,full=full);

# standard errors on the betas, based on an asymptotic Gaussian approx
# (these may be inflated for small n)

(standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,seqlik)

# a more reliable but slow way to get standard errors using a bootstrap

(standarderrorsbs,pvaluesbs,pvaluesbsnp,covmtxbs) = emerrorsbs(data,subs,x,X,h,betas,sigma,seqlik; parallel=parallel, full=full, nsamples=100, emtol=1e-3)

# another variant, by omitting the covariate of interest from the model

X2 = designmatrix([ones(NS), ones(NS),ones(NS), ones(NS), ones(NS), ones(NS)]);

betas2 = [1., 1., 1., 1., 1., 1. ];
sigma2 = [5.,5,5,5,1.0,5];
(betas2,sigma2,x2,l2,h2) = em(data,subs,X2,betas2,sigma2,seqlik; emtol=emtol, parallel=parallel, full=full);
lm(@formula(beta~cov),DataFrame(beta=x2[1,:],cov=cov))

## model selection/comparison/scoring

# laplace approximation to the aggregate log marginal likelihood of the whole dataset
# marginalized over the individual params

ll = lml(x,l,h)

# correct these for top-level parameters via bic or aic to compare between models

ibic(x,l,h,betas,sigma,NS*NT)
iaic(x,l,h,betas,sigma)

# unbiased per-subject marginal likelihoods by cross validation, for eg SPM_BMS

liks = loocv(data,subs,x,X,betas,sigma,seqlik;emtol=1e-3, parallel=parallel, full=full)
