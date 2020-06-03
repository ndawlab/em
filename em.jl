# julia EM model fitting, Nathaniel Daw 8/2019

#### basic fitting routine

function em(data,subs,X,betas,sigma::Vector,likfun; emtol=1e-4, parallel=false, startx = [], maxiter=100, quiet=false, full=false)
	if full
		return em(data,subs,X,betas,Matrix(Diagonal(sigma)),likfun; emtol=emtol, parallel=parallel, startx = startx, maxiter=maxiter, full=full, quiet=quiet)
	else
		return em(data,subs,X,betas,Diagonal(sigma),likfun; emtol=emtol, parallel=parallel, startx = startx, maxiter=maxiter, full=full, quiet=quiet)
	end
end

function em(data,subs,X,betas,sigma,likfun; emtol=1e-4, parallel=false, startx = [], maxiter=100, quiet=false, full=false)
	nsub = size(X,1)
    nparam = size(betas,2)

	newparams = packparams(betas,sigma)
	
	betas = betas
	sigma = sigma
	iter = 0

	# allocate memory for the subject-level results

	if (parallel)
		h = SharedArray{Float64,3}((nparam,nparam,nsub), pids=workers())
		l = SharedArray{Float64,1}((nsub), pids=workers())
		x = SharedArray{Float64,2}((nsub,nparam), pids=workers())
	else
		h = zeros(nparam,nparam,nsub)
		l = zeros(nsub)
		x = zeros(nsub,nparam)
	end

	if isempty(startx) 
		x[:,:] = X * betas
	else
		x[:,:] = startx
	end

	while (true)
		oldparams = newparams
		(x, l, h) = estep(data,subs,x,x,l,h,X,betas,sigma,likfun,parallel=parallel) 
		(betas, sigma) = mstep(x,X,h,sigma,full=full)

		newparams = packparams(betas,sigma)

		iter += 1

		if (!quiet)
			if (@isdefined IJulia)
				IJulia.clear_output()
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

		if ((maximum(abs.((newparams-oldparams)./oldparams)) < emtol) | (iter > maxiter))
			return(betas,sigma,x,l,h)
		end
	end
end

### E and M steps

function estep(data,subs,startx,x,l,h,X,betas,sigma,likfun; parallel=false)
	nsub = length(subs)
	mus = X * betas
	nparam = size(mus,2)

	if (parallel)

		# parallel version stores results in shared memory between workers
		
		@sync @distributed for i = 1:nsub
			try  # errors in parallel code seem to disappear silently so this prints and throws them
				sub = subs[i];
				(l[i], x[i,:]) = optimizesubject((x) -> gaussianprior(x,mus[i,:],sigma,data[data[:,:sub] .== sub,:],likfun), startx[i,:]);
				
				hess = y -> ForwardDiff.hessian((x) -> gaussianprior(x,mus[i,:],sigma,data[data[:,:sub] .== sub,:],likfun), y);
	
				h[:,:,i] = inv(hess(x[i,:]));
			catch err
	 			error(err)
	 		end
	 	end

	else
		# single CPU version 
	
		for i = 1:nsub
			sub = subs[i]
			print(i,"..")
			
			(l[i], x[i,:]) = optimizesubject((x) -> gaussianprior(x,mus[i,:],sigma,data[data[:,:sub] .== sub,:],likfun), startx[i,:])
		
			hess = y -> ForwardDiff.hessian((x) -> gaussianprior(x,mus[i,:],sigma,data[data[:,:sub] .== sub,:],likfun), y)
		
			h[:,:,i] = inv(hess(x[i,:]))
		end
	end

	return(x,l,h)
end

function mstep(x,X,h,sigma;full=false)
	# this result from http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf
	# gives same output as more complicated Huys procedure, when design matrix complies with these conditions

	nsub = size(X,1)

	betas = inv(X' * X) * X' * x

	newsigma = x' * (I - X * inv(X'*X)*X') * x / nsub + dropdims(mean(h,dims=3),dims=3)

	if (~full)
		newsigma = Diagonal(newsigma)
	end

	if (det(newsigma)<0)
		println("Warning: sigma has negative determinant")
	else
		sigma = newsigma
	end

	if length(betas) == 1
		betas = betas[1]
	end

	return(betas,sigma)
end

function emcovmtx(data,subs,x,X,h,betas,sigma,likfun)
  	# compute covariance on the group level model parameters using missing information
    # this version from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.442.7750&rep=rep1&type=pdf
    # Tagare "A gentle introduction to the EM algorithm"

    nsub = size(X,1)
    nparam = size(betas,2)
    nreg = size(X,2)
    nbetas = prod(size(betas))

	prior = packparams(betas,sigma)

	h1beta = inv(kron(inv(X'*X), sigma))
	h1sigma = ForwardDiff.hessian(newsigma -> mobj(x,X,h,betas,newsigma,nparam), packsigma(sigma))
	h1 = zeros(length(prior),length(prior))
	h1[1:nbetas,1:nbetas] = h1beta
	h1[nbetas+1:end,nbetas+1:end] = h1sigma
	#h1 = h1 * (nsub-nreg) / nsub # bias correction

	h2 = ForwardDiff.hessian(newprior -> entropyterm(data,subs,x,X,h,betas,sigma,newprior,likfun), prior)

	return inv(h1-h2)[1:nbetas,1:nbetas]
end

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


#### model selection 

# aggregate / integrated measures

function lml(x,l,h)
	# this computes the laplace approximation to the log marginal likelihood.
	# this marginalizes over the subject level parameters but still
	# needs correcting for group-level parameters (see functions below)

	nparam = size(x,2)
	nsub = size(x,1)

	if any([det(h[:,:,i]) for i in 1:nsub] .< 0) 
		return NaN
	else
		return -nparam/2 * log(2*pi) * nsub + sum(l) - sum([log(det(h[:,:,i])) for i in 1:nsub])/2
	end
end

# aic & bic for group level parameters

function ibic(x,l,h,betas,sigma,ndata)
	return(lml(x,l,h) + length(packparams(betas,sigma))/2 * log(ndata))
end

function iaic(x,l,h,betas,sigma)
	return(lml(x,l,h) + length(packparams(betas,sigma)))
end

# model selection by leave one out cross validation (at the subject level)
# this uses laplace approximation to the marginal likelihood for each subject

function loocv(data,subs,startx,X,betas,sigma,likfun;emtol=1e-4,parallel=false, full=false, maxiter=100)
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
			(newbetas,newsigma,~,~,~) = em(data,loosubs,looX,betas,sigma,likfun; emtol=emtol, parallel=parallel, startx=loostartx, full=full, maxiter=maxiter, quiet=true)
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

	if (any([det(h[:,:,i]) for i in 1:nsub] .< 0) || det(sigma) < 0)
		return NaN
	else 
		return (sum([(
	    # MVN Log L (from Wikipedia) terms not involving subject level params x
	    -nparam/2*log(2*pi) - 1/2 * log(det(sigma)) -
	    # MVN LogL term involving x, in expectation over x from Eq 7a in Roweis cheat sheet
	    1/2 * ((x[sub,:]-mu[sub,:])' * inv(sigma) * (x[sub,:]-mu[sub,:]) + tr(inv(sigma) * h[:,:,sub] )) +
	    # entropy of hidden variables (from Wikipedia)
	    nparam/2*log(2*pi*exp(1)) + 1/2 * log(det(h[:,:,sub])))
	    for sub in 1:nsub])[1]
	    # expected LL for the observations
	    - lml(x,l,h))
    end
end

