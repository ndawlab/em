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

function informationmatrixsigma(sigma, nsub)
	# this computes the sub block of the complete information matrix for the sigma parameters
    nparam = size(sigma, 1)
    sigmainv = inv(sigma)
    
    # Define row-wise unrolling rule (corresponding to packparams)
	if isdiag(sigma)
        unique_pairs = [(i, i) for i in 1:nparam] # Only diagonal elements]
    else
        unique_pairs = [(i, j) for i in 1:nparam for j in i:nparam] # Full upper-triangular
    end
    num_params = length(unique_pairs)
    
	# Pre-build the basis derivative matrices (V_k) for each unique parameter
    V_matrices = map(unique_pairs) do (i, j)
        V = zeros(nparam, nparam)
        V[i, j] = 1.0
        V[j, i] = 1.0 # Enforces symmetry for off-diagonals automatically
        return V
    end
    
    # Compute the Information Matrix using the trace formula
    I_ΣΣ = [
        (nsub / 2) * tr(sigmainv * V_matrices[row] * sigmainv * V_matrices[col])
        for row in 1:num_params, col in 1:num_params
    ]
    
    return I_ΣΣ
end


function missing_information(x, X, h, betas, sigma)
	# this computes the missing information using a Laplace approx to the Louis (1982) formula
	# this fully analytic form is pretty messy due to the unrolling of the sigma parameters
	# (the autograd version is much easier to read but egregious)

    nsub = size(X, 1)
    nreg = size(X, 2)
    nparam = size(x, 2)
    
    is_diagonal = (typeof(sigma) <: Diagonal)
    ncov_params = is_diagonal ? nparam : Int(nparam * (nparam + 1) / 2)
    ntheta = (nreg * nparam) + ncov_params
    
    I_missing = zeros(ntheta, ntheta)
    sigma_inv = inv(sigma)
    
    for i in 1:nsub
        w_hat = x[i, :]         
        V_i = h[:, :, i]         
        
        mu_i = betas' * X[i, :]        
        residual = w_hat - mu_i        
        
        J_i = zeros(ntheta, nparam)
        row_idx = 1
        
        # 1. MEAN BLOCK (B)
        for r in 1:nreg
            for p in 1:nparam
                J_i[row_idx, :] = X[i, r] .* sigma_inv[:, p]
                row_idx += 1
            end
        end
        
        # 2. COVARIANCE BLOCK (Sigma)
        if is_diagonal
            # --- Diagonal Case ---
            for j in 1:nparam
                E_j = zeros(nparam, nparam)
                E_j[j, j] = 1.0
                J_i[row_idx, :] = sigma_inv * E_j * sigma_inv * residual
                row_idx += 1
            end
        else
            # --- Full Symmetric Case (Upper Triangular Row-by-Row) ---
            for r_cov in 1:nparam
                for c_cov in r_cov:nparam
                    
                    if r_cov == c_cov
                        # Diagonal element: appears exactly once in the matrix
                        E_jc = zeros(nparam, nparam)
                        E_jc[r_cov, r_cov] = 1.0
                        J_i[row_idx, :] = sigma_inv * E_jc * sigma_inv * residual
                    else
                        # Off-diagonal element: maps symmetrically to TWO slots
                        # because altering the single parameter packs changes both coordinates!
                        E_jc = zeros(nparam, nparam)
                        E_jc[r_cov, c_cov] = 1.0
                        E_jc[c_cov, r_cov] = 1.0
                        
                        J_i[row_idx, :] = sigma_inv * E_jc * sigma_inv * residual
                    end
                    
                    row_idx += 1
                end
            end
        end
        
        # 3. Apply Louis's Sandwich Formula
        I_missing += J_i * V_i * J_i'
    end
    
    return I_missing
end


function emcovmtx(x,X,h,betas,sigma)
  	# compute covariance on the group level model parameters using missing information
    # this version from Tagare "A gentle introduction to the EM algorithm" eq 4.1

    nsub = size(X,1)
    nbetas = prod(size(betas))

	prior = packparams(betas,sigma)

	# the first term is the information matrix of the complete data likelihood 
	# (the "complete information")
	# it is block diagonal, with one block for the betas and one for the sigma
	# these are standard MVN formulas

	h1 = zeros(length(prior),length(prior))
	h1[1:nbetas,1:nbetas] = inv(kron(inv(X'*X), sigma))
	h1[nbetas+1:end,nbetas+1:end] = informationmatrixsigma(sigma,nsub)
	#h1 = h1 * (nsub-nreg) / nsub # bias correction

	# the second term is the missing information
	h2 = missing_information(x, X, h, betas, sigma)

	return inv(h1-h2)[1:nbetas,1:nbetas]
end




"""
    emerrors(x,X,h,betas,sigma)
Compute approximate standard errors for the coefficients from a model estimated by `em()`

# Arguments
- `x`: the per-subject parameter estimates from `em()`
- `X`: the design matrix, with a column per group-level predictor and a row per subject
- `h`: the per-subject inverse hessians from `em()`
- `betas`: the estimated group-level coefficients from `em()`
- `sigma`: the estimated group-level variance vector or covariance matrix from `em()`

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
function emerrors(x,X,h,betas,sigma)
    nsub = size(X,1)
    nreg = size(X,2)
    nparam = size(betas,2)

 	covmtx = emcovmtx(x,X,h,betas,sigma)

	ses = sqrt.([diag(covmtx)[i] .< 0 ? NaN : diag(covmtx)[i] for i in 1:length(diag(covmtx))])

	# dof from helwig notes
	pvalues = 2*ccdf.(TDist(nparam*(nsub - nreg - 1)), abs.(vec(betas')) ./ ses)
	#pvalues = 2*ccdf.(Normal(0,1),abs.(vec(betas')) ./ ses)

	return (ses,pvalues,covmtx)
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

	return -nparam/2 * log(2*pi) * nsub + sum(l) - sum([logdet(h[:,:,i]) for i in 1:nsub if incsub[i]])/2
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

