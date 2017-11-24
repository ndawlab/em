# julia EM model fitting, Nathaniel Daw 7/2016

function optimizesubjectjulia(likfun, startx)
	a=Optim.optimize(likfun,startx,method=:bfgs,ftol=1e-32,grtol=1e-10,autodiff=true)
	#	a=Optim.optimize(likfun,startx,method=:bfgs,autodiff=true)
	#display(a)

	return(a.f_minimum,a.minimum)
end


function optimizesubjectpython(likfun, startx)
	# this uses python's optimization function which seems to work a little better than julia's
	a = so.minimize(likfun, startx, method="L-BFGS-B", jac = (x->ForwardDiff.gradient(likfun,x)))
	#println(a["message"])

	return((a["fun"],a["x"])::Tuple{Float64,Array{Float64,1}})
end
	

function gaussianprior(params,mu,sigma,data,likfun)
	d = length(params)

    lp = -d/2 * log(2*pi) - 1/2 * log(det(sigma)) - 1/2 * (params - mu)' * inv(sigma) * (params - mu)
	 
	nll = likfun(params, data)
	
	return (nll - lp[1])
end


function computemeans(X,betas)
	nparam = size(X,1)
	nsub = size(X,3)

	mu = zeros(typeof(betas[1]),nparam,nsub)
	for i = 1:nsub
		mu[:,i] = X[:,:,i] * betas;
	end

	return mu
end


# create 3-D design matrix from cell array of per-parameter design matrices
function designmatrix(X)
	nparam = length(X)
	nsub = size(X[1],1)
	nreg = sum([length(X[i][1,:]) for i in 1:nparam])
