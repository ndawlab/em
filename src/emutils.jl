# julia EM model fitting, Nathaniel Daw 8/2019

function optimizesubject(likfun, startx)
	#a = optimize(likfun, startx, NewtonTrustRegion(); autodiff=:forward)
	a = optimize(likfun, startx, LBFGS(); autodiff=:forward)

	return(a.minimum,a.minimizer)
end

# replace the above function with this one to use python optimizer
# also need to uncomment code in EM.jl
# and run single threaded (eg unset JULIA_NUM_THREADS)
# because this is not thread safe

#function optimizesubject(likfun, startx)
#	# this uses python's optimization function which used to work a little better than julia's
#	# but is not thread safe
#	a = so.minimize(likfun, startx, method="L-BFGS-B", jac = (x->ForwardDiff.gradient(likfun,x)))
#	#println(a["message"])
#
#	return((a["fun"],a["x"])::Tuple{Float64,Array{Float64,1}})
#end

function gaussianprior(params,mu,sigma,data,likfun)
	d = length(params)

    lp = -d/2 * log(2*pi) - 1/2 * log(det(sigma)) - 1/2 * (params - mu)' * inv(sigma) * (params - mu)
	 
	nll = likfun(params, data)
	
	return (nll - lp[1])
end

# utilities for packing and unpacking the top level betas and sigmas into a vector (for hessians etc)

flatten(a::Array{T,1}) where T = any(x->isa(x,Array),a) ? flatten(vcat(map(flatten,a)...)) : a
flatten(a::Array{T}) where T = reshape(a,prod(size(a)))
flatten(a)=a

function packparams(betas,sigma)
	# we transpose b here and below so it reads out columnwise
	if typeof(betas) == Float64
		return([betas;packsigma(sigma)])
	else
		return [vec(betas'); packsigma(sigma)]
	end
end

function packsigma(sigma)
	l = size(sigma,1)
	return flatten([sigma[i,i:l] for i in 1:l])
end

function packsigma(sigma::Diagonal)
	return diag(sigma)
end


# ugliness to unpack the vector of betas and sigmas back into the appropriate matrices

function unpackparams(prior,nreg,nparam)
	# transposed so it all goes columnwise
	betas = reshape(prior[1:(nreg*nparam)],nparam,nreg)'

	sigma = unpacksigma(prior[(nreg*nparam+1):end], nparam)
	return (betas,sigma)
end

function unpacksigma(sigmapacked,nparam)
	if (length(sigmapacked) == nparam)
		sigma = Diagonal(sigmapacked)
	else
		sigma = zeros(typeof(sigmapacked[1]),nparam,nparam)
		n = 1

		for i = 1:nparam
			sigma[i,i] = sigmapacked[n]
			n += 1
			for j = i+1:nparam
				sigma[i,j] = sigmapacked[n]
				sigma[j,i] = sigmapacked[n]
				n += 1
			end
		end
	end

	return sigma
end

# Use this instead of "max" in the bellman equation lookahead so that
# gradients are better behaved
 
function softmaximum(a,b)
	p=1/(1+exp(-5*(a-b)))
	return(p * a + (1-p) * b)
end
