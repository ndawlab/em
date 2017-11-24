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
