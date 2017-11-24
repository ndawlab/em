# julia EM model fitting, Nathaniel Daw 9/2017


###### setup 

parallel = true # Run on multiple CPUs. If you are having trouble, set parallel = false: easier to debug
full = false    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM

if (parallel)
	# only run this once
	addprocs()
end


using DataFrames
using ForwardDiff
using PyCall
using Distributions
using GLM

# this loads the packages needed -- the @everywhere makes sure they 
# available on all CPUs 
@everywhere PyCall.@pyimport scipy.optimize as so

# change this to where you keep the code
@everywhere directory = "/users/ndaw/Dropbox/expts/julia\ em/git/em"

@everywhere include("$directory/em.jl");
@everywhere include("$directory/common.jl");
@everywhere include("$directory/likfuns.jl")

###### Q learning example

# simulate some  qlearning data
NS = 20;
NT = 200;
NP = 2;

params = zeros(NS,NP);

cov = randn(NS); # simulated between-subeject variable, e.g. age or IQ
cov = cov - mean(cov);
# subject level parameters
params[:,1] = 1 + 0.5 * cov + 0.5 * randn(NS);
params[:,2] = randn(NS);

c = zeros(Int64,NS*NT);
r = zeros(Int64,NS*NT);
