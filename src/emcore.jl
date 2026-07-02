# julia EM model fitting, Nathaniel Daw 12/2021

#### basic fitting routines
"""
    em(model::EMModel; nparam=nothing, startbetas=nothing, startsigma=nothing, emtol=1e-3, startx=[], maxiter=100, quiet=10, full=false)

Fit a hierarchical model using Expectation-Maximization (EM).

# Arguments
- `model::EMModel`: Object housing the dataset, subject list, design matrix `X`, and likelihood function.
- `nparam::Int`: (Optional) Number of subject-level parameters. If provided (and starting values are omitted), group coefficients `betas` are initialized to zero and variances `sigma` to one.
- `startbetas`: Starting points for group-level coefficients (number of predictors x number of parameters).
- `startsigma`: Starting point for group-level covariance matrix (given as a matrix or a vector of diagonal elements).
- `emtol=1e-3`: Stopped point tolerance for relative change in parameters.
- `startx=X*betas`: Starting points for per-subject parameters (subjects x parameters).
- `maxiter=100`: Maximum EM iterations.
- `quiet=10`: Print progress updates every N iterations (0: never).
- `full=false`: Use a full (vs. diagonal) group-level covariance.

# Returns
- Returns an `EMFit` struct with the following fields:
  - `betas::Matrix{Float64}`: The estimated group-level coefficients (predictors x parameters).
  - `sigma::Union{Diagonal, Matrix}`: The estimated group-level variance vector (represented as Diagonal) or covariance matrix.
  - `x::Matrix{Float64}`: The per-subject parameter estimates (subjects x parameters).
  - `l::Vector{Float64}`: The per-subject negative log-likelihoods.
  - `h::Array{Float64, 3}`: The per-subject inverse Hessians.
  - `model::EMModel`: Reference to the original model definition.

The `EMFit` object can be destructured as:
```julia
betas, sigma, x, l, h = fit
```

---

    em(data, subs, X, betas, sigma, likfun; emtol=1e-3, startx=[], maxiter=100, quiet=10, full=false)

Backward-compatible positional arguments version of `em`.
"""

# Fit a model using EM
function em(model::EMModel; nparam::Union{Int,Nothing}=nothing, startbetas=nothing, startsigma=nothing, emtol=1e-3, startx=[], maxiter=100, quiet=10, full=false)
    # Infer nparam if model.nparam == 0
    if model.nparam == 0
        actual_nparam = startbetas !== nothing ? size(startbetas, 2) :
                        (nparam !== nothing ? nparam :
                         (isempty(startx) ? 0 : size(startx, 2)))
        if actual_nparam == 0
            throw(ArgumentError("Must specify either nparam in EMModel constructor/argument, or starting values (startbetas/startx) to determine nparam."))
        end
        # Reconstruct model with correct nparam
        model = EMModel(model.data, model.subs, model.X, actual_nparam, model.nsub, model.nreg, model.likfun, model.reg_names, model.param_names)
    end

    nsub = model.nsub
    nreg = model.nreg
    nparam_val = model.nparam

    # Determine starting values
    if startbetas === nothing && startsigma === nothing
        s_betas = zeros(nreg, nparam_val)
        s_sigma = ones(nparam_val)
    elseif startbetas !== nothing && startsigma !== nothing
        s_betas = startbetas
        s_sigma = startsigma
    else
        throw(ArgumentError("Must specify both startbetas and startsigma, or neither."))
    end

    # Handle vector/diagonal/matrix covariance format conversions
    if typeof(s_sigma) <: Vector
        s_sigma = full ? Matrix(Diagonal(s_sigma)) : Diagonal(s_sigma)
    elseif full && typeof(s_sigma) <: Diagonal
        s_sigma = Matrix(s_sigma)
    elseif !full && !(typeof(s_sigma) <: Diagonal) && isdiag(s_sigma)
        s_sigma = Diagonal(s_sigma)
    end

    # Allocate memory for subject-level results
    h = zeros(nparam_val, nparam_val, nsub)
    l = zeros(nsub)
    x = zeros(nsub, nparam_val)

    if isempty(startx)
        x[:, :] = model.X * s_betas
    else
        x[:, :] = startx
    end

    if (Threads.nthreads() == 1)
        @warn "Not running in parallel. Please set JULIA_NUM_THREADS environment variable & restart."
    end

    fit = EMFit(s_betas, s_sigma, x, l, h, model)
    newparams = packparams(fit.betas, fit.sigma)
    iter = 0

    while (true)
        oldparams = newparams
        estep!(fit)
        mstep!(fit)

        newparams = packparams(fit.betas, fit.sigma)
        iter += 1

        done = ((maximum(abs.((newparams - oldparams) ./ oldparams)) < emtol) | (iter > maxiter))
        if ((quiet > 0) && (done || (iter % quiet == 0)))
            if isdefined(Main, :IJulia) && Main.IJulia.inited
                Main.IJulia.clear_output()
            end
            println("\niter: ", iter)
            println("betas: ", round.(fit.betas, digits=2))
            if isdiag(fit.sigma)
                println("sigma: ", round.(diag(fit.sigma), digits=2))
            else
                println("sigma: ", round.(fit.sigma, digits=2))
            end
            println("free energy: ", round(freeenergy(fit), digits=6))
            println("change: ", round.(abs.(newparams - oldparams) ./ oldparams, digits=6))
            println("max: ", round.(maximum(abs.((newparams - oldparams) ./ oldparams)), digits=6))
        end

        if done
            return fit
        end
    end
end


### E and M steps

function estep!(f::EMFit)
    m = f.model
    mus = m.X * f.betas

    inv_sigma = inv(f.sigma)
    logdet_sigma = logdet(f.sigma)

    Threads.@threads for i = 1:m.nsub
        sub = m.subs[i]

        fitfun = (params) -> gaussianprior(params, view(mus, i, :), inv_sigma, logdet_sigma, view(m.data, m.data.sub .== sub, :), m.likfun)

        (f.l[i], min_x) = optimizesubject(fitfun, f.x[i, :])
        f.x[i, :] = min_x

        hess = y -> ForwardDiff.hessian(fitfun, y)
        f.h[:, :, i] = inv(hess(min_x))
    end
    nothing
end

function mstep!(f::EMFit)
    # this result from http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf
    m = f.model

    f.betas = inv(m.X' * m.X) * m.X' * f.x
    is_diagonal = (typeof(f.sigma) <: Diagonal)

    proj = I - m.X * inv(m.X' * m.X) * m.X'
    newsigma = f.x' * proj * f.x / m.nsub + dropdims(mean(f.h, dims=3), dims=3)

    if (det(newsigma) < 0)
        println("Warning: sigma has negative determinant")
    else
        if is_diagonal
            f.sigma = Diagonal(diag(newsigma))
        else
            f.sigma = newsigma
        end
    end

    nothing
end

#### functions related to error bars

function informationmatrixsigma(sigma, nsub)
    # this computes the sub block of the complete information matrix for the sigma parameters
    # it is pretty ugly due to the unrolling of the sigma parameters.
    # the version that calls autograd is much easier to read but gratuitous
    nparam = size(sigma, 1)
    sigmainv = inv(sigma)
    A = sigmainv

    if isdiag(sigma)
        unique_pairs = [(i, i) for i in 1:nparam] # Only diagonal elements
    else
        unique_pairs = [(i, j) for i in 1:nparam for j in i:nparam] # Full upper-triangular
    end
    num_params = length(unique_pairs)

    I_ΣΣ = zeros(num_params, num_params)
    for row in 1:num_params
        (i, j) = unique_pairs[row]
        for col in 1:num_params
            (k, l) = unique_pairs[col]

            val = 0.0
            if i == j && k == l
                val = A[i, k]^2
            elseif i != j && k == l
                val = 2.0 * A[i, k] * A[j, k]
            elseif i == j && k != l
                val = 2.0 * A[i, k] * A[i, l]
            else # i != j && k != l
                val = 2.0 * (A[i, k] * A[j, l] + A[i, l] * A[j, k])
            end

            I_ΣΣ[row, col] = (nsub / 2) * val
        end
    end

    return I_ΣΣ
end


function missing_information(fit::EMFit)
    # this computes the missing information using a Laplace approx to the Louis (1982) formula
    # this fully analytic form is pretty messy due to the unrolling of the sigma parameters
    # (the autograd version is much easier to read but gratuitous)
    m = fit.model

    is_diagonal = (typeof(fit.sigma) <: Diagonal)
    ncov_params = is_diagonal ? m.nparam : Int(m.nparam * (m.nparam + 1) / 2)
    ntheta = (m.nreg * m.nparam) + ncov_params

    I_missing = zeros(ntheta, ntheta)
    sigma_inv = inv(fit.sigma)

    for i in 1:m.nsub
        w_hat = fit.x[i, :]
        V_i = fit.h[:, :, i]

        mu_i = fit.betas' * m.X[i, :]
        residual = w_hat - mu_i

        J_i = zeros(ntheta, m.nparam)
        row_idx = 1

        # 1. MEAN BLOCK (B)
        for r in 1:m.nreg
            for p in 1:m.nparam
                J_i[row_idx, :] = m.X[i, r] .* sigma_inv[:, p]
                row_idx += 1
            end
        end

        # 2. COVARIANCE BLOCK (Sigma)
        if is_diagonal
            # --- Diagonal Case ---
            for j in 1:m.nparam
                E_j = zeros(m.nparam, m.nparam)
                E_j[j, j] = 1.0
                J_i[row_idx, :] = sigma_inv * E_j * sigma_inv * residual
                row_idx += 1
            end
        else
            # --- Full Symmetric Case (Upper Triangular Row-by-Row) ---
            for r_cov in 1:m.nparam
                for c_cov in r_cov:m.nparam

                    if r_cov == c_cov
                        # Diagonal element: appears exactly once in the matrix
                        E_jc = zeros(m.nparam, m.nparam)
                        E_jc[r_cov, r_cov] = 1.0
                        J_i[row_idx, :] = sigma_inv * E_jc * sigma_inv * residual
                    else
                        # Off-diagonal element: maps symmetrically to TWO slots
                        # because altering the single parameter packs changes both coordinates!
                        E_jc = zeros(m.nparam, m.nparam)
                        E_jc[r_cov, c_cov] = 1.0
                        E_jc[c_cov, r_cov] = 1.0

                        J_i[row_idx, :] = sigma_inv * E_jc * sigma_inv * residual
                    end

                    row_idx += 1
                end
            end
        end

        # 3. Apply Louis's Formula
        I_missing += J_i * V_i * J_i'
    end

    return I_missing
end


function emcovmtx(fit::EMFit)
    # compute covariance on the group level model parameters using missing information
    # this version from Tagare "A gentle introduction to the EM algorithm" eq 4.1

    m = fit.model
    nbetas = prod(size(fit.betas))

    prior = packparams(fit.betas, fit.sigma)

    # the first term is the information matrix of the complete data likelihood 
    # (the "complete information")
    # it is block diagonal, with one block for the betas and one for the sigma
    # these are standard MVN formulas

    h1 = zeros(length(prior), length(prior))
    h1[1:nbetas, 1:nbetas] = inv(kron(inv(m.X' * m.X), fit.sigma))
    h1[nbetas+1:end, nbetas+1:end] = informationmatrixsigma(fit.sigma, m.nsub)

    # the second term is the missing information
    h2 = missing_information(fit)

    return inv(h1 - h2)[1:nbetas, 1:nbetas]
end

"""
    emerrors(fit::EMFit; reg_names=nothing, param_names=nothing)

Compute approximate standard errors, p-values, and covariance matrix for the coefficients of a fitted model. This is the primary, recommended API.

# Arguments
- `fit::EMFit`: The fitted model object returned by `em()`.

# Returns
- Returns an `EMErrors` struct with the following fields:
  - `ses::Vector{Float64}`: Standard errors per coefficient.
  - `pvalues::Vector{Float64}`: p-values for the null hypothesis that each coefficient = 0.
  - `covmtx::Matrix{Float64}`: The covariance matrix over the coefficients.
  - `fit::EMFit`: Reference to the parent model fit object.

The `EMErrors` object can be destructured as:
```julia
ses, pvals, covmtx = errors
```
- Printing is formatted as a significance regression table showing regressor and parameter names, estimates, standard errors, t-values, p-values, and significance stars.

---

    emerrors(x, X, h, betas, sigma)

Backward-compatible positional arguments version of `emerrors`.
"""
function emerrors(fit::EMFit)
    m = fit.model

    covmtx = emcovmtx(fit)

    ses = sqrt.([diag(covmtx)[i] .< 0 ? NaN : diag(covmtx)[i] for i in 1:length(diag(covmtx))])

    # dof from helwig notes
    pvalues = 2 * ccdf.(TDist(m.nparam * (m.nsub - m.nreg - 1)), abs.(vec(fit.betas')) ./ ses)

    return EMErrors(ses, pvalues, covmtx, fit)
end



#### functions related to model selection 

# aggregate / integrated measures
"""
    lml(fit::EMFit)

Compute the Laplace approximation to the log-marginal likelihood of the dataset for a fitted model. This is the primary, recommended API.
This marginalizes over the subject-level parameters but note that it is conditional on (not corrected for overfitting due to) the estimated group-level parameters.

---

    lml(x, l, h)

Backward-compatible positional arguments version of `lml`.
"""
function lml(fit::EMFit)
    m = fit.model

    incsub = [det(fit.h[:, :, i]) > 0 for i in 1:m.nsub]

    if any(.!incsub)
        n = sum(.!incsub)
        println("Warning: Omitting from LML $n subjects with non-invertible Hessian")
    end

    return -m.nparam / 2 * log(2 * pi) * m.nsub + sum(fit.l) - sum([logdet(fit.h[:, :, i]) for i in 1:m.nsub if incsub[i]]) / 2
end


function ibic(fit::EMFit, ndata)
    return lml(fit) + length(packparams(fit.betas, fit.sigma)) / 2 * log(ndata)
end


function iaic(fit::EMFit)
    return lml(fit) + length(packparams(fit.betas, fit.sigma))
end


# model selection by leave one out cross validation (at the subject level)
# this uses laplace approximation to the marginal likelihood for each subject

"""
    loocv(fit::EMFit; emtol=1e-3, full=false, maxiter=100)

Compute per-subject leave-one-subject-out predictive likelihood scores under a fitted model.
Scores are computed from cross-validated group-level parameters, with each subject left out, and using a Laplace approximation to marginalize the subject-level parameters. 

# Arguments
- `fit::EMFit`: The fitted model object returned by `em()`.
- `emtol=1e-3`: Stopping point tolerance for relative change in parameters.
- `full=false`: Use a full (vs. diagonal) group-level covariance.
- `maxiter=100`: Maximum EM iterations per-subject.

---

    loocv(data, subs, startx, X, betas, sigma, likfun; emtol=1e-3, full=false, maxiter=100)

Backward-compatible positional arguments version of `loocv`.
"""
function loocv(f::EMFit; emtol=1e-3, full=false, maxiter=100)
    m = f.model

    liks = zeros(m.nsub)

    print("Subject: ")

    for i = 1:m.nsub
        sub = m.subs[i]

        print(i, "..")

        if (i == 1)
            loosubs = m.subs[2:end]
            looX = m.X[2:end, :]
            loostartx = f.x[2:end, :]
        elseif (i == m.nsub)
            loosubs = m.subs[1:end-1]
            looX = m.X[1:end-1, :]
            loostartx = f.x[1:end-1, :]
        else
            loosubs = [m.subs[1:i-1]; m.subs[i+1:end]]
            looX = m.X[[1:i-1; i+1:end], :]
            loostartx = f.x[[1:i-1; i+1:end], :]
        end

        try
            loo_model = EMModel(m.data, loosubs, looX, m.nparam, m.likfun)
            loo_fit = em(loo_model; startbetas=f.betas, startsigma=f.sigma, emtol=emtol, startx=loostartx, full=full, maxiter=maxiter, quiet=0)
            newmu = loo_fit.betas' * m.X[i, :]

            liks[i] = heldoutsubject_laplace(newmu, loo_fit.sigma, m.data[m.data[:, :sub].==sub, :], m.likfun; startx=f.x[i, :])
        catch err
            println(err)
            liks[i] = NaN
        end
    end

    return (liks)
end


function heldoutsubject_laplace(mu, sigma, data, likfun; startx=mu)
    nparam = length(mu)

    inv_sigma = inv(sigma)
    logdet_sigma = logdet(sigma)
    fitfun = (x) -> gaussianprior(x, mu, inv_sigma, logdet_sigma, data, likfun)

    (lik, params) = optimizesubject(fitfun, startx)

    hess = ForwardDiff.hessian(fitfun, params)

    lik = -nparam / 2 * log(2 * pi) + lik + log(det(hess)) / 2

    return (lik)
end


# attempt to compute the free energy expression as given in Gharamani EM slides

function freeenergy(f::EMFit)
    m = f.model

    mu = m.X * f.betas

    if (det(f.sigma) < 0)
        return NaN
    end

    incsub = [det(f.h[:, :, i]) > 0 for i in 1:m.nsub]

    inv_sigma = inv(f.sigma)
    logdet_sigma = logdet(f.sigma)

    val = sum(
        # MVN Log L (from Wikipedia) terms not involving subject level params x
        -m.nparam / 2 * log(2 * pi) - 0.5 * logdet_sigma -
        # MVN LogL term involving x, in expectation over x from Eq 7a in Roweis cheat sheet
        0.5 * (dot(f.x[sub, :] - mu[sub, :], inv_sigma, f.x[sub, :] - mu[sub, :]) + tr(inv_sigma * f.h[:, :, sub]))
        # entropy of hidden variables (from Wikipedia)
        # these terms also appear in LML below but I think they belong twice
        + m.nparam / 2 * log(2 * pi * exp(1)) + 0.5 * logdet(f.h[:, :, sub])
        for sub in 1:m.nsub if incsub[sub]
    )

    # expected LL for the observations
    return val - lml(f)
end
