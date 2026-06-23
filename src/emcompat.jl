# wrappers for old positional arguments interface

function em(data,subs,X,betas,sigma::Vector,likfun; emtol=1e-3, startx = [], maxiter=100, quiet=10, full=false)
    model = EMModel(data, subs, X, size(betas, 2), likfun)
    return em(model; startbetas=betas, startsigma=sigma, emtol=emtol, startx=startx, maxiter=maxiter, quiet=quiet, full=full)
end

function em(data,subs,X,betas,sigma,likfun; emtol=1e-3, startx = [], maxiter=100, quiet=10, full=false)
    model = EMModel(data, subs, X, size(betas, 2), likfun)
    return em(model; startbetas=betas, startsigma=sigma, emtol=emtol, startx=startx, maxiter=maxiter, quiet=quiet, full=full)
end

function emerrors(x,X,h,betas,sigma; reg_names=nothing, param_names=nothing)
	nsub = size(X,1)
	nparam = size(x,2)
	
	betas_f64 = Matrix{Float64}(betas)
	x_f64 = Matrix{Float64}(x)
	X_f64 = Matrix{Float64}(X)
	h_f64 = Array{Float64, 3}(h)
	
	s_sigma = if typeof(sigma) <: Vector
		Diagonal(Vector{Float64}(sigma))
	elseif typeof(sigma) <: Diagonal
		Diagonal(Vector{Float64}(sigma.diag))
	else
		Matrix{Float64}(sigma)
	end

	dummy_model = EMModel(DataFrame(), 1:nsub, X_f64, nparam, () -> ())
	dummy_fit = EMFit(betas_f64, s_sigma, x_f64, zeros(nsub), h_f64, dummy_model)
	return emerrors(dummy_fit; reg_names=reg_names, param_names=param_names)
end

function lml(x,l,h)
	nparam = size(x,2)
	nsub = size(x,1)
	dummy_model = EMModel(DataFrame(), 1:nsub, zeros(nsub, 1), nparam, () -> ())
	dummy_fit = EMFit(zeros(1, nparam), Diagonal(ones(nparam)), x, l, h, dummy_model)
	return lml(dummy_fit)
end

function ibic(x,l,h,betas,sigma,ndata)
	nparam = size(x, 2)
	nsub = size(x, 1)
	dummy_model = EMModel(DataFrame(), 1:nsub, zeros(nsub, size(betas, 1)), nparam, () -> ())
	dummy_fit = EMFit(betas, sigma, x, l, h, dummy_model)
	return ibic(dummy_fit, ndata)
end

function iaic(x,l,h,betas,sigma)
	nparam = size(x, 2)
	nsub = size(x, 1)
	dummy_model = EMModel(DataFrame(), 1:nsub, zeros(nsub, size(betas, 1)), nparam, () -> ())
	dummy_fit = EMFit(betas, sigma, x, l, h, dummy_model)
	return iaic(dummy_fit)
end


function loocv(data,subs,startx,X,betas,sigma,likfun; emtol=1e-3, full=false, maxiter=100)
	nsub = size(X,1)
	nparam = size(startx,2)
	dummy_model = EMModel(data, subs, X, nparam, likfun)
	dummy_fit = EMFit(betas, sigma, startx, zeros(nsub), zeros(nparam, nparam, nsub), dummy_model)
	return loocv(dummy_fit; emtol=emtol, full=full, maxiter=maxiter)
end
