# julia EM model fitting, Nathaniel Daw 7/2016

#### basic fitting routine

function em(data,subs,X,betas,sigma::Vector,likfun; emtol=1e-4, parallel=false, startx = [], full=false, maxiter=100, quiet=false)
	if full
		return em(data,subs,X,betas,diagm(sigma),likfun; emtol=emtol, parallel=parallel, startx = startx, maxiter=maxiter, full=full, quiet=quiet)
	else
		return em(data,subs,X,betas,Diagonal(diagm(sigma)),likfun; emtol=emtol, parallel=parallel, startx = startx, maxiter=maxiter, full=full, quiet=quiet)
	end
end

function em(data,subs,X,betas,sigma,likfun; emtol=1e-4, parallel=false, startx = [], full=false, maxiter=100, quiet=false)
	nsub = size(X,3)
	nparam = size(X,1)

	if isempty(startx) 
		x = computemeans(X,betas)
	else
		x = startx
	end

	newparams = packparams(betas,sigma)
	betas = betas
	sigma = sigma
	iter = 0

	while (true)
		oldparams = newparams
		(x, l, h) = estep(data,subs,x,X,betas,sigma,likfun,parallel=parallel) 
		(betas, sigma) = mstep(x,X,h,sigma,full=full)
		newparams = packparams(betas,sigma)

		iter += 1

		if (!quiet)
			if (isdefined(:IJulia))
				IJulia.clear_output()
			end
			println("\niter: ", iter)
			println("betas: ", round.(betas,2))
			if isdiag(sigma)
				println("sigma: ", round.(diag(sigma),2))
				#println("cv: ", round(sqrt(diag(sigma))./betas,2))
			else
				println("sigma: ", round.(sigma,2))
				#println("cv: ", round(sqrt(sigma)./betas,2))
			end
			#println("free energy: ", round(freeenergy(x,l,h,X,betas,sigma),6))
			println("change: ", round.(abs.(newparams-oldparams)./oldparams,6))
			println("max: ", round.(maximum(abs.(newparams-oldparams)./oldparams),6))
		end	

		if ((maximum(abs.(newparams-oldparams)./oldparams) < emtol) | (iter > maxiter))
			return(betas,sigma,x,l,h)
		end
	end
end

### E and M steps

function estep(data,subs,startx,X,betas,sigma,likfun; parallel=false)
	nparam = size(X,1)
	
	mus = computemeans(X,betas)

	# a little optimization for subject level bootstrap:
	# don't repeat e step for subjects occurring more than once

	uniquesubs = unique(subs)
	nsub = length(subs)
	nusub = length(uniquesubs)

	if nusub < nsub
		subsfit = uniquesubs
		mus = mus[:,[find(subs .== uniquesubs[i])[1] for i in 1:nusub]]
		startx = startx[:,[find(subs .== uniquesubs[i])[1] for i in 1:nusub]]
	else
		subsfit = subs
	end

	if (parallel)

		# parallel version stores data in shared memory between workers
		
		h = SharedArray{Float64}((nparam,nparam,nusub), pids=workers())
		l = SharedArray{Float64}((nusub), pids=workers())
		x = SharedArray{Float64}((nparam,nusub), pids=workers())

		@sync @parallel for i = 1:nusub
			try  # errors in parallel code seem to disappear silently so this prints and throws them
				sub = subsfit[i];
				(l[i], x[:,i]) = optimizesubjectpython((x) -> gaussianprior(x,mus[:,i],sigma,data[data[:sub] .== sub,:],likfun), startx[:,i]);
				
				hess = y -> ForwardDiff.hessian((x) -> gaussianprior(x,mus[:,i],sigma,data[data[:sub] .== sub,:],likfun), y);
	
				h[:,:,i] = inv(hess(x[:,i]));
			catch err
	 			error(err)
	 		end
	 	end

	else

		h = zeros(nparam,nparam,nusub)
		l = zeros(nusub)
		x = zeros(nparam,nusub)
	
		for i = 1:nusub
			sub = subsfit[i]
			print(i,"..")
			
			(l[i], x[:,i]) = optimizesubjectpython((x) -> gaussianprior(x,mus[:,i],sigma,data[data[:sub] .== sub,:],likfun), startx[:,i])
		
			hess = y -> ForwardDiff.hessian((x) -> gaussianprior(x,mus[:,i],sigma,data[data[:sub] .== sub,:],likfun), y)
		
			h[:,:,i] = inv(hess(x[:,i]))
		end
	end

	if (nusub < nsub)
		# copy subjects appearing more htan once

		indices = [find(uniquesubs .== subs[i])[1] for i in 1:nsub]
		h = h[:,:,indices]
		l = l[indices]
		x = x[:,indices]
	end

	return(x,l,h)
end

function mstep(x,X,h,sigma;full=false)
	nsub = size(X,3)

	if (~full || isdiag(X[:,:,1]))
		# note the expression below (with isigma) still applies, but those terms cancel in these cases
		# so are omitted here

		betas = inv(mean([X[:,:,i]' * X[:,:,i] for i in 1:nsub])) * mean([X[:,:,i]' * x[:,i] for i in 1:nsub])
		mu = computemeans(X,betas)
		newsigma = (x-mu)*(x-mu)'/nsub + squeeze(mean(h,3),3)
		if (~full)
			newsigma = Diagonal(newsigma)
		end
	else
		# in this case even the M step is iterative (like iteratively reweighted LS) as per Quentin's code
		# since the covariance and the subject-level estimates depend on one another

		iter = 0
		newsigma = sigma
		while (true)
			isigma = inv(newsigma)
			betas = inv(mean([X[:,:,i]'*isigma*X[:,:,i] for i in 1:nsub])) * mean([X[:,:,i]' * isigma * x[:,i] for i in 1:nsub])
			mu = computemeans(X,betas)
			newsigma = (x-mu)*(x-mu)'/nsub + squeeze(mean(h,3),3)

			iter += 1

			if ( (iter > 1 && maximum(abs.(betas-oldbetas)./oldbetas) < 1e-4) | (iter > 100))
				#println("finished m step in $iter iterations\n")
				break
			end

			oldbetas = betas
		end
	end

	if (det(newsigma)<0)
		println("Warning: sigma has negative determinant")
	else
		sigma = newsigma
	end

	return(betas,sigma)
end


### functions related to error bars, p values and covariance matrices

function emcovmtx(data,subs,x,X,h,betas,sigma,likfun)
  	# compute covariance on the group level model parameters using Eq 6
    # from Oakes, J R Stat Soc B 1999 

	prior = packparams(betas,sigma)
	nprior = length(prior)

	h1 = ForwardDiff.hessian(newprior -> mobj(x,X,h,newprior), prior) 
	h2 = ForwardDiff.hessian(oldpriornewprior -> emobj(data,subs,x,X,likfun,oldpriornewprior),[prior;prior])[1:nprior,nprior+1:end]

	return inv(h1+h2)
end

function embscovmtx(data,subs,x,X,h,betas,sigma,likfun; parallel=false, full=false, nsamples=100, emtol=1e-3)

	#  bootstrap at the top level to estimate covariance

	nprior = length(packparams(betas,sigma))
	nbetas = size(X,2)
	nparam = size(X,1)

	estimates = zeros(nprior,nsamples)
	print("Bootstrap iteration: ")

	for i = 1:nsamples
		newsubis = sample(1:length(subs),length(subs))
		newsubs = subs[newsubis]
		newX = X[:,:,newsubis]
		newx = x[:,newsubis]

		if (isdefined(:IJulia))
			IJulia.clear_output()
		end

		print("$i ")

		(newbetas,newsigma,~,~,~) = em(data,newsubs,newX,betas,sigma,likfun; emtol=emtol, parallel=parallel, startx=newx, full=full, quiet=true)

		estimates[:,i] = packparams(newbetas,newsigma)
	end
	println()

	return(StatsBase.cov(estimates'),estimates)
end

function embscovmtxfast(data,subs,x,X,h,betas,sigma,likfun; full=false, nsamples=100)

	# quick approximations to a real top-level bootstrap, either using just an M step or an M-fakeE-M cycle.

	nprior = length(packparams(betas,sigma))
	nbetas = size(X,2)
	nparam = size(X,1)
 
	estimates = zeros(nprior,nsamples)

	for i = 1:nsamples
		newsubis = sample(1:length(subs),length(subs))
		newsubs = subs[newsubis]
		newX = X[:,:,newsubis]
		newx = x[:,newsubis]
		newh = h[:,:,newsubis]

		# XXX - can't update sigma for the resampling here, mstep has to work it out
		(newbetas,newsigma) = mstep(newx,newX,newh,sigma,full=full)

		# comment the following two lines out to just use one M step
		(newx,newh) = fakeestep(data,newsubs,newx,newX,newbetas,newsigma,likfun)
		(newbetas,newsigma) = mstep(newx,newX,newh,newsigma,full=full)

		estimates[:,i] = packparams(newbetas,newsigma)
	end

	return(StatsBase.cov(estimates'),estimates)
end

function fakeestep(data,subs,startx,X,betas,sigma,likfun)

	# this is a fake e step based on a Gauss/Newton approximation

	nsub = size(X,3)
	nparam = size(X,1)
	
	mus = computemeans(X,betas)

	newh = zeros(nparam,nparam,nsub)
	newx = zeros(nparam,nsub)
	
	for i = 1:nsub
		sub = subs[i]
	
		gsub = ForwardDiff.gradient((x) -> likfun(x,data[data[:sub] .== sub,:]),startx[:,i]) - inv(sigma) * (mus[:,i] - startx[:,i])
		hsub = ForwardDiff.hessian((x) -> likfun(x,data[data[:sub] .== sub,:]),startx[:,i]) + inv(sigma) 
		
		newx[:,i] = startx[:,i] - inv(hsub) * gsub

		# note that h is the same everywhere under a 2nd order approximation, i.e. we don't extrapolate a different value for newh at newx vs startx
		# though we could in principle use a third order derivative to do so 

		newh[:,:,i] = inv(hsub)
	end

	return(newx,newh)
end

function emcontrast(betas,contrast,covmtx)
	value = betas' * contrast
	covvar = contrast' * covmtx * contrast
	covse = covvar < 0 ? NaN:sqrt(covvar)
	pvalue = 2*ccdf(Normal(0,1),abs(value) ./ covse)

	return(value[],covse[],pvalue[])
end

function emerrors(data,subs,x,X,h,betas,sigma,likfun)
    nbetas = size(X,2)

	covmtx = emcovmtx(data,subs,x,X,h,betas,sigma,likfun)

	ses = sqrt.([diag(covmtx)[i] .< 0 ? NaN:diag(covmtx)[i] for i in 1:nbetas])
	pvalues = 2*ccdf(Normal(0,1),abs.(betas) ./ ses)

	return (ses,pvalues,covmtx[1:nbetas,1:nbetas])
end

function emerrorsbs(data,subs,x,X,h,betas,sigma,likfun; parallel=false, full=false, nsamples=100, emtol=1e-3)
    nbetas = size(X,2)

	(covmtx,samples) = embscovmtx(data,subs,x,X,h,betas,sigma,likfun; parallel=parallel, full=full, nsamples=nsamples, emtol=emtol)
 	ses = sqrt.(diag(covmtx)[1:nbetas])
	pvalues = 2*ccdf(Normal(0,1),abs.(betas) ./ ses)
	pvaluesnp = [mean(abs.(samples[j,:]-betas[j]) .> abs.(betas[j])) for j in 1:nbetas ]

	return (ses,pvalues,pvaluesnp,covmtx[1:nbetas,1:nbetas])
end

function emerrorsbsfast(data,subs,x,X,h,betas,sigma,likfun; full=false, nsamples=100)
    nbetas = size(X,2)

	(covmtx,samples) = embscovmtxfast(data,subs,x,X,h,betas,sigma,likfun; full=full, nsamples=nsamples)
	ses = sqrt.(diag(covmtx)[1:nbetas])
	pvalues = 2*ccdf(Normal(0,1),abs.(betas) ./ ses)
	pvaluesnp = [mean(abs.(samples[j,:]-betas[j]) .> abs.(betas[j])) for j in 1:nbetas ]

	return (ses,pvalues,pvaluesnp,covmtx[1:nbetas,1:nbetas])
end

function mobj(x,X,h,prior)
	# this is the objective function for the m step (for computing the information matrix)
	nsub = size(X,3)
	nbetas = size(X,2)
	nparam = size(X,1)
	
	(betas,sigma) = unpackparams(prior,nparam,nbetas)

	mu = computemeans(X,betas)
 
 	# eq 7a from Roweis Gaussian cheat sheet

	return -sum([-nparam/2*log(2*pi) - 1/2 * log(det(sigma)) - 1/2 * ((x[:,sub]-mu[:,sub])' * inv(sigma) * (x[:,sub]-mu[:,sub]) + trace(inv(sigma) * h[:,:,sub] )) for sub in 1:nsub])[1]
end

function emobj(data,subs,startx,X,likfun,oldnewprior)
	# this is the objective function for the m step with an e step preceding it
	# (for computing the information matrix)
	# this substitutes a single newton update as a differentiable approx to
	# the optimization in the e step
	# I have also combined old and new prior in a single vector so the mixed seond derivative can be computed as a sub-block of the hessian

	len = Int(length(oldnewprior)/2)
	oldprior = oldnewprior[1:len]
	newprior = oldnewprior[len+1:end]
	nsub = size(X,3)
	nparam = size(X,1)
	nbetas = size(X,2)

	(oldbetas,oldsigma) = unpackparams(oldprior,nparam,nbetas)
	
	# the following code (until return) is an approximation to:
	#(newx,l,newh) = estep(data,subs,startx,X,oldbetas,oldsigma2,likfun,parallel=parallel)

	mus = computemeans(X, oldbetas)
	
	newx = zeros(typeof(oldprior[1]),nparam,nsub)
	newh = zeros(typeof(oldprior[1]),nparam,nparam,nsub);
	
	for i = 1:nsub
		sub = subs[i]
		
		gsub = ForwardDiff.gradient((x) -> likfun(x,data[data[:sub] .== sub,:]),startx[:,i]) - inv(oldsigma) * (mus[:,i] - startx[:,i])
		hsub = ForwardDiff.hessian((x) -> likfun(x,data[data[:sub] .== sub,:]),startx[:,i]) + inv(oldsigma) 
		
		newx[:,i] = startx[:,i] - inv(hsub) * gsub

		# note that h is the same everywhere under a 2nd order approximation, i.e. we don't extrapolate a different value for newh at newx vs startx
		# though we could in principle use a third order derivative to do so 

		newh[:,:,i] = inv(hsub)
	end
		
	return mobj(newx,X,newh,newprior)
end


#### model selection 

# aggregate / integrated measures

function lml(x,l,h)
	# this computes the laplace approximation to the log marginal likelihood.
	# this marginalizes over the subject level parameters but still
	# needs correcting for group-level parameters (see functions below)

	nparam = size(x,1)
	nsub = size(x,2)

	return -nparam/2 * log(2*pi) * nsub + sum(l) - sum([log(det(h[:,:,i])) for i in 1:nsub])/2
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

function loocv(data,subs,startx,X,betas,sigma,likfun;emtol=1e-4,parallel=false, full=false)
	nsub = size(X,3)

	liks = zeros(nsub)
	
	print("Subject: ")

	for i = 1:nsub
		sub = subs[i]

		print(i,"..")

		if (i==1)
			loosubs = subs[2:end]
			looX = X[:,:,2:end]
			loostartx = startx[:,2:end]
		elseif (i==nsub)
			loosubs = subs[1:end-1]
			looX = X[:,:,1:end-1]
			loostartx = startx[:,1:end-1]
		else
			loosubs = [subs[1:i-1];subs[i+1:end]]
			looX = X[:,:,[1:i-1;i+1:end]]
			loostartx = startx[:,[1:i-1;i+1:end]]
		end

		try
			(newbetas,newsigma,~,~,~) = em(data,loosubs,looX,betas,sigma,likfun; emtol=emtol, parallel=parallel, startx=loostartx, full=full, quiet=true)
			newmu = X[:,:,i] * newbetas;

			liks[i] = heldoutsubject_laplace(newmu,newsigma,data[data[:sub] .== sub,:],likfun;startx = startx[:,i])
		catch err
	 		println(err)
	 		liks[i] = NaN
	 	end
	end

	return(liks)
end

function heldoutsubject_laplace(mu, sigma, data, likfun; startx = mu)
	nparam = length(mu)

	(lik, params) = optimizesubjectpython((x) -> gaussianprior(x,mu,sigma,data,likfun), startx);
	
	hess = ForwardDiff.hessian((x) -> gaussianprior(x,mu,sigma,data,likfun),params);

	lik = -nparam/2 * log(2*pi) + lik + log(det(hess))/2
	
	return(lik)
end

function heldoutsubject_sample(mu, sigma, data, likfun; nsamples=1000)
	samples = rand(MvNormal(mu,PDMats.PDMat(sigma,cholfact(Hermitian(sigma)))), nsamples)
	ll = zeros(nsamples)

	for j = 1:nsamples
		ll[j] = -likfun(samples[:,j], data)
	end
	
	return(-log(sum(exp.(ll))) + log(nsamples))
end


# attempt to compute the free energy expression as given in Gharamani EM slides
function freeenergy(x,l,h,X,betas,sigma) 
	nsub = size(X,3)
	nbetas = size(X,2)
	nparam = size(X,1)

	mu = computemeans(X,betas)

	return (sum([(
	# MVN Log L (from Wikipedia) terms not involving subject level params x
	-nparam/2*log(2*pi) - 1/2 * log(det(sigma)) -
	# MVN LogL term involving x, in expectation over x from Eq 7a in Roweis cheat sheet
	1/2 * ((x[:,sub]-mu[:,sub])' * inv(sigma) * (x[:,sub]-mu[:,sub]) + trace(inv(sigma) * h[:,:,sub] )) +
	# entropy of hidden variables (from Wikipedia)
	nparam/2*log(2*pi*e) + 1/2 * log(det(h[:,:,sub])))
	for sub in 1:nsub])[1]
	# expected LL for the observations
	- lml(x,l,h))
end
