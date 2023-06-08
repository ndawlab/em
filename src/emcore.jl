# julia EM model fitting, Nathaniel Daw 12/2021

#### basic fitting routines
"""
    em(data,subs,X,startbetas,startsigma,likfun; optional named arguments)
Fit a model using expectation-maximization.

# Arguments
- `data:DataFrame`: A data frame containing the data to be fit. Should have a column `sub` indicating the subject for each observation.
- `subs`: a vector or range of subjects to be considered (e.g. unique(data.sub))
- `X`: the design matrix, with a column per group-level predictor and a row per subject
- `startbetas`: starting points for group-level coefficients (number of predictors x number of parameters)
- `startsigma`: starting point for group-level variance vector or covariance matrix
- `likfun`: likelihood function: takes a dataframe and a vector of parameters, returns negative log likelihood
- `emtol=1e-3`: stopping point tolerance for relative change in parameters
- `full=false`: use a full (vs. diagonal) group-level covariance
- `maxiter=100`: maximum EM iterations
- `quiet=10`: print updates every N iterations (or 0: never)
- `startx=X*betas`: starting points for per-subject parameters

# Returns
returns `(betas,sigma,x,l,h)`
- `betas`: the estimated group-level coefficients
- `sigma`: the estimated group-level vector of variances (if `full=false`) or covariance matrix (if `full=true`)
- `x`: the per-subject parameters
- `l`: the per-subject likelihoods
- `h`: the per-subject inverse Hessians
"""
function em(data,subs,X,betas,sigma::Vector,likfun; emtol=1e-3, startx = [], maxiter=100, quiet=10, full=false)
	if full
		return em(data,subs,X,betas,Matrix(Diagonal(sigma)),likfun; emtol=emtol, startx = startx, maxiter=maxiter, full=full, quiet=quiet)
	else
		return em(data,subs,X,betas,Diagonal(sigma),likfun; emtol=emtol, startx = startx, maxiter=maxiter, full=full, quiet=quiet)
	end
end

function em(data,subs,X,betas,sigma,likfun; emtol=1e-3, startx = [], maxiter=100, quiet=10, full=false)
	nsub = size(X,1)
    nparam = size(betas,2)

	newparams = packparams(betas,sigma)
	
	betas = betas
	sigma = sigma
	iter = 0

	# allocate memory for the subject-level results

	h = zeros(nparam,nparam,nsub)
	l = zeros(nsub)
	x = zeros(nsub,nparam)

	if isempty(startx) 
		x[:,:] = X * betas
	else
		x[:,:] = startx
	end

	if (Threads.nthreads() == 1)
		@warn "Not running in parallel. Please set JULIA_NUM_THREADS environment variable & restart."
	end

	while (true)
		oldparams = newparams
		estep!(data,subs,x,x,l,h,X,betas,sigma,likfun) 
		(betas, sigma) = mstep(x,X,h,sigma)

		newparams = packparams(betas,sigma)

		iter += 1
		done =  ((maximum(abs.((newparams-oldparams)./oldparams)) < emtol) | (iter > maxiter))
		if ((quiet > 0) && (done || (iter % quiet == 0)))
			if isdefined(Main, :IJulia) && Main.IJulia.inited
				Main.IJulia.clear_output()
			end
			println("\niter: ", iter)
			println("betas: ", round.(betas,digits=2))
			if isdiag(sigma)
				println("sigma: ", round.(diag(sigma),digits=2))
			else
				println("sigma: ", round.(sigma,digits=2))
			end
			println("free energy: ", round(freeenergy(x,l,h,X,betas,sigma),digits=6))
			println("change: ", round.(abs.(newparams-oldparams)./oldparams,digits=6))
			println("max: ", round.(maximum(abs.((newparams-oldparams)./oldparams)),digits=6))
		end	

		if done
			return(betas,sigma,x,l,h)
		end
	end
end

# experimental function to generate starting points for em()

function eminits(data,subs,X,betas,sigma::Vector,likfun;nstarts=10)
	nsub = size(X,1)
    nparam = size(betas,2)

	x = zeros(nsub,nparam)
	l = zeros(nsub) .+ Inf

	startx = zeros(nstarts,nparam)
	for j = 1:nstarts
		#startx[j,:] = rand(MvNormal(vec((X*betas)[1,:]),PDMats.PDMat((Matrix(Diagonal(sigma))),cholesky(Hermitian(Matrix(Diagonal(sigma)))))))
		startx[j,:] = rand(MvNormal(vec((X*betas)[1,:]),Diagonal(sigma)))		
	end

	Threads.@threads for i = 1:nsub
		sub = subs[i];
		fitfun = (x) -> gaussianprior(x,(X*betas)[1,:],Diagonal(sigma),view(data,data.sub .== sub,:),likfun)

		for j = 1:nstarts
			(ll,xx) = optimizesubject(fitfun, startx[j,:]);		
			if ll < l[i]
				l[i] = ll
				x[i,:] = xx
			end
		end
	 end
	nothing

	return x
end



### E and M steps

function estep!(data,subs,startx,x,l,h,X,betas,sigma,likfun)
	nsub = length(subs)
	mus = X * betas
	nparam = size(mus,2)
		
	Threads.@threads for i = 1:nsub
		sub = subs[i];

		fitfun = (x) -> gaussianprior(x,mus[i,:],sigma,view(data,data.sub .== sub,:),likfun)

		(l[i], x[i,:]) = optimizesubject(fitfun, startx[i,:]);		
		hess = y -> ForwardDiff.hessian(fitfun, y);

		h[:,:,i] = inv(hess(x[i,:]));
	 end
	nothing
end

function mstep(x,X,h,sigma::Matrix)
	# this result from http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf
	# gives same output as more complicated Huys procedure, when design matrix complies with these conditions

	nsub = size(X,1)

	betas = inv(X' * X) * X' * x

	newsigma = x' * (I - X * inv(X'*X)*X') * x / nsub + dropdims(mean(h,dims=3),dims=3)


	if (det(newsigma)<0)
		println("Warning: sigma has negative determinant")
	else
		sigma = newsigma
	end

	#if length(betas) == 1
	#	betas = betas[1]
	#end

	return(betas,sigma)
end

function mstep(x,X,h,sigma::Diagonal)
	# for full = false

    (b,s) = mstep(x,X,h,Matrix(sigma))

	return(b,Diagonal(s))
end

#### functions related to error bars

function emcovmtx(data,subs,x,X,h,betas,sigma,likfun)
  	# compute covariance on the group level model parameters using missing information
    # this version from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.442.7750&rep=rep1&type=pdf
    # Tagare "A gentle introduction to the EM algorithm"
	#
	# this depends on derivatives only part of which I know analytically

    nsub = size(X,1)
    nparam = size(betas,2)
    nreg = size(X,2)
    nbetas = prod(size(betas))

	prior = packparams(betas,sigma)

	h1beta = inv(kron(inv(X'*X), sigma))
	# (in principle this is surely also analytic)
	h1sigma = ForwardDiff.hessian(newsigma -> mobj(x,X,h,betas,newsigma,nparam), packsigma(sigma))
	h1 = zeros(length(prior),length(prior))
	h1[1:nbetas,1:nbetas] = h1beta
	h1[nbetas+1:end,nbetas+1:end] = h1sigma
	#h1 = h1 * (nsub-nreg) / nsub # bias correction

	h2 = ForwardDiff.hessian(newprior -> entropyterm(data,subs,x,X,h,betas,sigma,newprior,likfun), prior)

	return inv(h1-h2)[1:nbetas,1:nbetas]
end

"""
    emerrors(data,subs,x,X,h,betas,sigma,likfun)
Compute approximate standard errors for the coefficients from a model estimated by `em()`

# Arguments
- `data::DataFrame`: the data
- `subs`: a vector or range of subjects to be considered (e.g. unique(data.sub))
- `x`: the per-subject parameter estimates from `em()`
- `X`: the design matrix, with a column per group-level predictor and a row per subject
- `h`: the per-subject inverse hessians from `em()`
- `betas`: the estimated group-level coefficients from `em()`
- `sigma`: the estimated group-level variance vector or covariance matrix from `em()`
- `likfun`: the likelihood function

# Returns
returns `(ses,pvalues,covmtx)`
- `ses`: standard errors per coefficient
- `pvalues`: p values for the null hypothesis that each coefficient = 0
- `covmtx`: the covariance matrix over the coeffients

Note that though `betas` is a matrix of predictors x parameters, these coefficients are reordered 
as a vector, `vec(betas')` for the purpose of this function. This determines the order of `ses`, 
`pvalues` and the arrangement of `covmtx`. You can rebuild them back into the shape of `betas``
using, e.g., `reshape(pvalues,size(betas'))'` 
"""
function emerrors(data,subs,x,X,h,betas,sigma,likfun)
    nsub = size(X,1)
    nreg = size(X,2)
    nparam = size(betas,2)

 	covmtx = emcovmtx(data,subs,x,X,h,betas,sigma,likfun)

	ses = sqrt.([diag(covmtx)[i] .< 0 ? NaN : diag(covmtx)[i] for i in 1:length(diag(covmtx))])

	# dof from helwing notes
	pvalues = 2*ccdf.(TDist(nparam*(nsub - nreg - 1)), abs.(vec(betas')) ./ ses)
	#pvalues = 2*ccdf.(Normal(0,1),abs.(vec(betas')) ./ ses)

	return (ses,pvalues,covmtx)
end

function mobj(x,X,h,betas,sigma,nparam)
	# this is the objective function for the m step (for computing the information matrix)
	# it consists of only the terms from the free energy that depend on sigma
	# (because I know the analytic part wrt betas)

    nsub = size(X,1)
    nreg = size(X,2)
	nparam = size(betas,2)

	sigma = unpacksigma(sigma,nparam)
	mu = X * betas
 
 	# eq 7a from Roweis Gaussian cheat sheet
	return -sum([-1/2 * log(det(sigma)) - 1/2 * ((x[sub,:]-mu[sub,:])' * inv(sigma) * (x[sub,:]-mu[sub,:]) + tr(inv(sigma) * h[:,:,sub] )) for sub in 1:nsub])[1]
end

function entropyterm(data,subs,x,X,h,oldbetas,oldsigma,prior,likfun)
	# this is the entropy term of the full likelihood, viewed as a function of the prior
	# for information matrix calculation 
	# retaining the terms that depend on the prior

	nsub = size(X,1)
    nreg = size(X,2)
    nparam = size(oldbetas,2)

    # construct a Gaussian approx to the subject level evidence

    (likx,likh) = subjectlikelihood(data,subs,x,X,h,oldbetas,oldsigma,likfun)
	
	# use this to construct a Gaussian approximation to the subject level posterior
	# given new top level params

	(betas,sigma) = unpackparams(prior,nreg,nparam)
	mu = X * betas
 
 	hnew = zeros(typeof(betas[1]),nparam,nparam,nsub)
	xnew = zeros(typeof(betas[1]),nsub,nparam)

 	for sub = 1:nsub
		hnew[:,:,sub] = inv(inv(sigma) + inv(likh[:,:,sub]))
		xnew[sub,:] = hnew[:,:,sub] * (inv(sigma) * mu[sub,:] + inv(likh[:,:,sub]) * likx[sub,:])
	end	
	
	# finally the expression: log p(x | newbetas, newsigma, data) in expectation over x,h
	# eq 7a from Roweis Gaussian cheat sheet
	return -sum([-1/2 * log(det(hnew[:,:,sub])) - 1/2 * ((x[sub,:]-xnew[sub,:])' * inv(hnew[:,:,sub]) * (x[sub,:]-xnew[sub,:]) + tr(inv(hnew[:,:,sub]) * h[:,:,sub] )) for sub in 1:nsub])[1]
end

function subjectlikelihood(data,subs,x,X,h,betas,sigma,likfun)
	# this produces a Gaussian approximation to the subject level likelihood
	# such that its Gaussian product with the prior gives the appropriate posterior
	# we use this to compute the missing information under the Laplace approximation

	nsub = size(X,1)
    nparam = size(betas,2)

	likh = zeros(typeof(betas[1]),nparam,nparam,nsub)
	likx = zeros(typeof(betas[1]),nsub,nparam)

	mus = X * betas

	for sub = 1:nsub
		likh[:,:,sub] = inv(inv(h[:,:,sub]) - inv(sigma))

		likx[sub,:] = likh[:,:,sub] * inv(h[:,:,sub]) * (x[sub,:] - h[:,:,sub] * inv(sigma) * mus[sub,:])
	end

	return(likx,likh)
end


#### functions related to model selection 

# aggregate / integrated measures
"""
    lml(x,l,h)
Computes a vector of per-subject log-marginal likelihoods for a model previously fit with `em()` (giving subject level parameters 
`x`, likelihoods `l`, and inverse hessians `h`). This marginalizes over the subject-level parameters using a Laplace approximation
but note that it is conditional on (not marginalized over or otherwise correcte for overfitting due to) the estimated 
group-level parameters.
"""
function lml(x,l,h)
	# this computes the laplace approximation to the log marginal likelihood.
	# this marginalizes over the subject level parameters but still
	# needs correcting for group-level parameters (see functions below)

	nparam = size(x,2)
	nsub = size(x,1)

	incsub = [det(h[:,:,i]) > 0 for i in 1:nsub]

	if any(.!incsub)
		n = sum(.!incsub)
		println("Warning: Omitting from LML $n subjects with non-invertible Hessian")
	end

	return -nparam/2 * log(2*pi) * nsub + sum(l) - sum([log(det(h[:,:,i])) for i in 1:nsub if incsub[i]])/2
end

# aic & bic for group level parameters

"""
    ibic(x,l,h,betas,sigma,ndata)

Compute the iBIC (integrated BIC; Huys et al. 2011) measure of model fit aggregated over subjects for a model 
previously fit by `em()``; this marginalizes subject level parameters using a Laplace approximation and then applies a BIC
penalty for group-level parameters. 

# Arguments
- `x`: the per-subject parameters
- `l`: the per-subject likelihoods
- `h`: the per-subject inverse Hessians
- `betas`: the group-level coefficients
- `sigma`: the group-level variance vector or covariance matrix
(... all returned from `em()`)
- `ndata`: the total number of datapoints on which the model was estimated (aggregated over all subjects)
"""
function ibic(x,l,h,betas,sigma,ndata)
	return(lml(x,l,h) + length(packparams(betas,sigma))/2 * log(ndata))
end

"""
    iaic(x,l,h,betas,sigma)

Compute the iAIC (integrated AIC; Huys et al. 2011) measure of model fit aggregated over subjects for a model 
previously fit by `em()``; this marginalizes subject level parameters using a Laplace approximation and then applies an AIC
penalty for group-level parameters. 
	
# Arguments
- `x`: the per-subject parameters
- `l`: the per-subject likelihoods
- `h`: the per-subject inverse Hessians
- `betas`: the group-level coefficients
- `sigma`: the group-level variance vector or covariance matrix
(... all returned from `em()`)
"""
function iaic(x,l,h,betas,sigma)
	return(lml(x,l,h) + length(packparams(betas,sigma)))
end

# model selection by leave one out cross validation (at the subject level)
# this uses laplace approximation to the marginal likelihood for each subject

"""
    loocv(data,subs,startx,X,betas,sigma,likfun;emtol=1e-3, full=false, maxiter=100)
Compute per-subject leave-one-subject-out predictive likelihood scores under a model previously fit using `em()`.
Scores are computed from cross-validated group-level parameters, with each subject left out, and using a Laplace
approximation to marginalize the subject-level parameters. 

# Arguments
- `data::DataFrame`: The data
- `subs`: vector or range of subjects
- `x`: starting points for re-estimating per-subject parameters (typically, per-subject estimates from `em()`)
- `X`: the design matrix
- `betas`: starting points for re-estimating group-level coeffients (typically, from `em()`)
- `sigma`: starting points for re-estimating group-level variances or covariance (typically, from `em()`)
- `emtol=1e-3`: stopping point tolerance for relative change in parameters
- `full=false`: use a full (vs. diagonal) group-level covariance
- `maxiter=100`: maximum EM iterations per-subject
""" 
function loocv(data,subs,startx,X,betas,sigma,likfun;emtol=1e-3, full=false, maxiter=100)
	nsub = size(X,1)

	liks = zeros(nsub)
	
	print("Subject: ")

	for i = 1:nsub
		sub = subs[i]

		print(i,"..")

		if (i==1)
			loosubs = subs[2:end]
			looX = X[2:end,:]
			loostartx = startx[2:end,:]
		elseif (i==nsub)
			loosubs = subs[1:end-1]
			looX = X[1:end-1,:]
			loostartx = startx[1:end-1,:]
		else
			loosubs = [subs[1:i-1];subs[i+1:end]]
			looX = X[[1:i-1;i+1:end],:]
			loostartx = startx[[1:i-1;i+1:end],:]
		end

		try
			(newbetas,newsigma,~,~,~) = em(data,loosubs,looX,betas,sigma,likfun; emtol=emtol, startx=loostartx, full=full, maxiter=maxiter, quiet=true)
			newmu = newbetas' * X[i,:]

			liks[i] = heldoutsubject_laplace(newmu,newsigma,data[data[:,:sub] .== sub,:],likfun;startx = startx[i,:])
		catch err
	 		println(err)
	 		liks[i] = NaN
	 	end
	end

	return(liks)
end

function heldoutsubject_laplace(mu, sigma, data, likfun; startx = mu)
	nparam = length(mu)

	(lik, params) = optimizesubject((x) -> gaussianprior(x,mu,sigma,data,likfun), startx);
	
	hess = ForwardDiff.hessian((x) -> gaussianprior(x,mu,sigma,data,likfun),params);

	lik = -nparam/2 * log(2*pi) + lik + log(det(hess))/2
	
	return(lik)
end


# attempt to compute the free energy expression as given in Gharamani EM slides

function freeenergy(x,l,h,X,betas,sigma) 
	nsub = size(x,1)
	nbetas = size(X,2)
	nparam = size(x,2)

	mu = X * betas

	if (det(sigma) < 0)
		return NaN
	end

	incsub = [det(h[:,:,i]) > 0 for i in 1:nsub]

	return (sum([(
	# MVN Log L (from Wikipedia) terms not involving subject level params x
	-nparam/2*log(2*pi) - 1/2 * log(det(sigma)) -
	# MVN LogL term involving x, in expectation over x from Eq 7a in Roweis cheat sheet
	1/2 * ((x[sub,:]-mu[sub,:])' * inv(sigma) * (x[sub,:]-mu[sub,:]) + tr(inv(sigma) * h[:,:,sub] )) 
	# entropy of hidden variables (from Wikipedia)
	# these terms also appear in LML below but I think they belong twice
	+ nparam/2*log(2*pi*exp(1)) + 1/2 * log(det(h[:,:,sub]))
	)
	for sub in 1:nsub if incsub[sub]])[1]
	# expected LL for the observations
	- lml(x,l,h))
    
end

