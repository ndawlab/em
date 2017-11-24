function sandwich(x,X,h,prior)
	nsub = size(X,3)
	gi = [-ForwardDiff.gradient(newprior -> mobjsub(x[:,sub],X[:,:,sub],h[:,:,sub],newprior),prior) for sub in 1:nsub]

	return sum([gi[sub] * gi[sub]' for sub in 1:nsub])
end

function mobjsub(xi,Xi,hi,prior)
	# this is the objective function for the m step for subject i (for computing the robust information matrix)
	nparam = size(Xi,1)
	nbetas = size(Xi,2)
	
	(betas,sigma) = unpackparams(prior,nparam,nbetas)

	mui = Xi * betas;

 	# eq 7a from Roweis Gaussian cheat sheet

	return (-nparam/2*log(2*pi) - 1/2 * log(det(sigma)) - 1/2 * ((xi-mui)' * inv(sigma) * (xi-mui) + trace(inv(sigma) * hi )) )[1] 
end


function testerrors(data,subs,x,X,h,betas,sigma,likfun)
    nbetas = size(X,2)
    nsub = size(X,3)

    factor = nsub/(nsub-2)
    df = nsub - 2;

    if (det(sigma) <= 0)
    	println(sigma)
	end

	covmtx = emcovmtx(data,subs,x,X,h,betas,sigma,likfun)

	if (all(diag(covmtx)[1:nbetas].>=0))
		ses = sqrt(diag(covmtx)[1:nbetas])
		z = abs(betas)./ses 
		pvaluesa = 2*ccdf(Normal(0,1),abs(betas) ./ ses)
		pvaluesa2 = 2*ccdf(TDist(df),abs(betas) ./ ses)
		ses = ses * factor
		pvaluesb = 2*ccdf(Normal(0,1),abs(betas) ./ ses)
		pvaluesb2 = 2*ccdf(TDist(df),abs(betas) ./ ses)
	else
		pvaluesa = NaN * zeros(nbetas)
		pvaluesa2 = NaN * zeros(nbetas)
		pvaluesb = NaN * zeros(nbetas)
		pvaluesb2 = NaN * zeros(nbetas)
		z = NaN * zeros(nbetas)
	end

	covmtx = emrobcovmtx(data,subs,x,X,h,betas,sigma,likfun)
	if (all(diag(covmtx)[1:nbetas].>=0))
		ses = sqrt(diag(covmtx)[1:nbetas])
		z2 = abs(betas)./ses 
		pvaluesc = 2*ccdf(Normal(0,1),abs(betas) ./ ses)
		pvaluesc2 = 2*ccdf(TDist(df),abs(betas) ./ ses)
		ses = ses * factor
		pvaluesd = 2*ccdf(Normal(0,1),abs(betas) ./ ses)
		pvaluesd2 = 2*ccdf(TDist(df),abs(betas) ./ ses)
	else
		pvaluesc = NaN * zeros(nbetas)
		pvaluesc2 = NaN * zeros(nbetas)
		pvaluesd = NaN * zeros(nbetas)
		pvaluesd2 = NaN * zeros(nbetas)
		z2 = NaN * zeros(nbetas)
	end

	return (pvaluesa,pvaluesb,pvaluesc,pvaluesd,pvaluesa2,pvaluesb2,pvaluesc2,pvaluesd2,z,z2)
end



function emrobcovmtx(data,subs,x,X,h,betas,sigma,likfun)
  	# compute covariance on the group level model parameters using Eq 6
    # from Oakes, J R Stat Soc B 1999 

    # this version uses a huber/white robust ("sandwich") estimator 

	prior = packparams(betas,sigma)
	nprior = length(prior)

	h1 = ForwardDiff.hessian(newprior -> mobj(x,X,h,newprior),prior) 
	h2 = ForwardDiff.hessian(oldpriornewprior -> emobj(data,subs,x,X,likfun,oldpriornewprior),[prior;prior])[1:nprior,nprior+1:end]
	B = sandwich(x,X,h,prior)
	A = inv(h1+h2)
	return A*B*A
end




function loocvsamples(data,subs,startx,X,betas,sigma,likfun;emtol=1e-4,parallel=false, full=false, nsamples=2000)
	nsub = size(X,3)
	nparam = size(X,1)

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
			print("optimizing..")
			(newbetas,newsigma,~,~,~) = em(data,loosubs,looX,betas,sigma,likfun; emtol=emtol, parallel=parallel, startx=loostartx, full=full, quiet=true)
			newmu = X[:,:,i] * newbetas;

			println("sampling..")

			samples = randn(nparam,nsamples);

			a = eigfact(newsigma)
			A = a[:vectors] * sqrt(diagm(a[:values]))
	
			datasub = data[data[:sub] .== subs[i],:]

			ll = zeros(nsamples)

			for j = 1:nsamples
				ll[j] = -likfun(newmu + A * samples[:,j], datasub)
			end
	
			liks[i] = logsumexp(ll) - log(nsamples)
		catch err
	 		println(err)
	 		liks[i] = NaN
	 	end
	end

	return(-liks)
end



function lmlsamples(betas, sigma, data,subs,X,likfun; nsamples=2000, samples = [])
	# this computes the sample-based approximation to the log marginal likelihood.
	# (if you wanted to take derivs etc of this, youd need to pass in a matrix of samples
	# to keep the samples constant. the function is written in terms of unit random
	# samples to make this possible)

	nparam = size(X,1)
	nsub = size(X,3)

	# make the samples

	if isempty(samples)
		samples = zeros(nparam,nsamples,nsub);
		for i = 1:nsub
  			samples[:,:,i] = randn(nparam,nsamples);
		end
	else
		nsamples = size(samples,2)
	end

	mu = computemeans(X,betas)

	ll = zeros(typeof(betas[1]),nsamples,nsub)

	# for transforming unit normal random variables to have cov sigma

	a = eigfact(sigma)
	A = a[:vectors] * sqrt.(diagm(a[:values]))
	
	for i = 1:nsub
		print(i,"..")
		datasub = data[data[:sub] .== subs[i],:]

		for j = 1:nsamples
			ll[j,i] = -likfun(mu[:,i] + A * samples[:,j,i], datasub)
		end
	end			
	
	ll = sum(log(sum(exp.(ll),1))) - nsub * log(nsamples)

	return -ll
end

#mutable struct emstruct
#	data
#	subs
#	X
#	x
#	betas
#	sigma
#	likfun
#	parallel
#	full
#	maxiter
#	quiet
#end


