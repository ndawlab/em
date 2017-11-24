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
	
	X2 = zeros(nparam,nreg,nsub)
	for i =1 :nsub
		n = 1
		for j = 1:nparam
   	  		l = length(X[j][i,:])
         	X2[j,n:(n+l-1),i] = X[j][i,:]
         	n += l 
     	end
	end

	return(X2)
end

# utilities for packing and unpacking the top level betas and sigmas into a vector (for hessians etc)

flatten{T}(a::Array{T,1}) = any(x->isa(x,Array),a)? flatten(vcat(map(flatten,a)...)): a
flatten{T}(a::Array{T}) = reshape(a,prod(size(a)))
flatten(a)=a

function unpackparams(prior,nparam,nbetas)
	# ugliness to unpack the vector of betas and sigmas back into the appropriate matrices

	betas = prior[1:nbetas]

	if (length(prior) == (nbetas + nparam) )
		sigma = Diagonal(prior[nbetas+1:end])
	else
		sigmas = prior[nbetas+1:end]
		sigma = zeros(typeof(prior[1]),nparam,nparam)
		n = 1
		
		for i = 1:nparam
			sigma[i,i] = sigmas[n]
			n += 1
			for j = i+1:nparam
				sigma[i,j] = sigmas[n]
				sigma[j,i] = sigmas[n]
				n += 1
			end
		end
	end

	return (betas,sigma)
end

function packparams(betas,sigma)
	#return [Array{Float64,1}(flatten(betas));sigma]
	l = size(sigma,1)
	return [betas; flatten([sigma[i,i:l] for i in 1:l])]
end

function packparams(betas,sigma::Diagonal)
	return [betas; diag(sigma)]
end

# Use this instead of "max" in the bellman equation lookahead so that
# gradients are better behaved
 
function softmaximum(a,b)
	p=1/(1+exp(-5*(a-b)))
	return(p * a + (1-p) * b)
end

# make sure gradients can deal with logsumexp
# import StatsBase.logsumexp
# logsumexp{T<:ForwardDiff.ForwardDiffNumber}(x::T, y::T) = x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
