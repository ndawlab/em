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
