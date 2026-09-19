# julia EM model fitting, Nathaniel Daw 12/2021

# Fit a model using EM
"""
    em(model::EMModel; nparam=nothing, startbetas=nothing, startsigma=nothing, emtol=1e-3, startx=[], maxiter=100, quiet=10, full=false, skewcorrect=false, skewcorrect_maxcorr=1.0)

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
- `skewcorrect=false`: Apply a third-derivative (skewness) correction to the Laplace approximation at every E-step, rather than plain mode/Hessian. Corrects for per-subject posteriors that are meaningfully non-Gaussian (e.g. when a jointly-estimated parameter takes an extreme value that degrades identifiability of another parameter). Costs roughly 3x the runtime of the plain E-step; falls back to the plain Laplace estimate for any subject/iteration where the correction itself isn't trustworthy (see `skewcorrect_maxcorr`).
- `skewcorrect_maxcorr=1.0`: Only used when `skewcorrect=true`. Safety threshold (in Laplace-SD units) above which a skewness correction is discarded in favor of the plain Laplace estimate for that subject-iteration, on the grounds that a correction that large signals the underlying perturbative approximation has broken down.

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

    em(data, subs, X, betas, sigma, likfun; emtol=1e-3, startx=[], maxiter=100, quiet=10, full=false, skewcorrect=false, skewcorrect_maxcorr=1.0)

Backward-compatible positional arguments version of `em`.
"""
function em(model::EMModel; nparam::Union{Int,Nothing}=nothing, startbetas=nothing, startsigma=nothing, emtol=1e-3, startx=[], maxiter=100, quiet=10, full=false, skewcorrect=false, skewcorrect_maxcorr=1.0)
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
    total_hessian_fallback = 0
    total_skew_fallback = 0

    while (true)
        oldparams = newparams
        fallback = skewcorrect ? estep_skewcorrect!(fit; maxcorr=skewcorrect_maxcorr) : estep!(fit)
        total_hessian_fallback += fallback.hessian_fallback
        total_skew_fallback += fallback.skew_fallback
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
            if quiet > 0 && total_hessian_fallback > 0
                println("Hessian fallback (non-PD Laplace Hessian, used group-level prior covariance instead): ",
                        total_hessian_fallback, " / ", iter * nsub, " subject-iterations")
            end
            if skewcorrect && quiet > 0
                println("skew-correction fallback: ", total_skew_fallback, " / ", iter * nsub, " subject-iterations")
            end
            return fit
        end
    end
end


### E and M steps

# The Hessian of gaussianprior at the mode is Hessian(nll) + inv_sigma: the
# Gaussian prior term contributes inv_sigma to the curvature everywhere, so it
# always supplies a positive floor. A non-positive-definite result therefore
# specifically means the *data* likelihood's own curvature overwhelmed that
# floor -- either because the optimizer hasn't actually reached a local
# minimum (a genuine local min of a smooth function always has a PSD
# Hessian), or because the subject's likelihood is so weakly informative that
# even the best nearby point still has a bad direction. If so, fall back to 
# inv_sigma itself: the mathematically correct limiting Hessian for a subject
# whose data contributes no trustworthy curvature information, i.e. their
# posterior reverts to exactly the group-level prior, N(mu_i, sigma).
function safe_laplace_hessian(fitfun, xhat, inv_sigma)
    H = ForwardDiff.hessian(fitfun, xhat)
    if isposdef(Symmetric(H))
        return H, false
    else
        return inv_sigma, true
    end
end

# Fits a single subject via the mode + (optionally skewness-corrected)
# Laplace approximation. Shared by estep!/estep_skewcorrect! (training) and
# heldoutsubject_laplace (loocv's held-out evaluation). Returns
# (x, l, H, hessian_fallback, skew_fallback), where H is always
# positive-definite (so f.h[:,:,i] = inv(H) always is too).
#
# The skewness correction is a first-order perturbative expansion, so it can
# misbehave for subjects where it's least trustworthy. Two checks guard
# against that: if the correction is large relative to the Laplace spread
# itself (Mahalanobis norm under H), or if the Hessian at the corrected
# point isn't positive-definite (it isn't guaranteed to be, since the
# corrected point need not be a local minimum), the correction is discarded
# and the plain Laplace mode/Hessian is returned instead.
function subject_laplace_fit(fitfun, startx, inv_sigma; skewcorrect=false, maxcorr=1.0)
    d = length(startx)
    (l0, xhat) = optimizesubject(fitfun, startx)

    H, fellback = safe_laplace_hessian(fitfun, xhat, inv_sigma)
    if fellback || !skewcorrect
        return (xhat, l0, H, fellback, false)
    end
    Hinv = inv(H)

    # third-derivative tensor of Q(x) = -log(unnormalized posterior) at
    # the mode, via one more layer of AD on top of the Hessian
    hessfun = y -> ForwardDiff.hessian(fitfun, y)
    J = ForwardDiff.jacobian(y -> vec(hessfun(y)), xhat)
    T3 = reshape(J, d, d, d)

    # classical first-order skewness correction to the posterior mean:
    # dx = -1/2 * Hinv * S,  S_j = sum_{k,kk} T3[k,kk,j] * Hinv[k,kk]
    S = zeros(d)
    for j in 1:d
        s = 0.0
        for k in 1:d, kk in 1:d
            s += T3[k, kk, j] * Hinv[k, kk]
        end
        S[j] = s
    end
    dx = -0.5 .* (Hinv * S)
    dxsize = sqrt(dot(dx, H, dx))  # size of the correction in Laplace-SD units

    xcorr = xhat .+ dx
    Hcorr = Symmetric(ForwardDiff.hessian(fitfun, xcorr))

    if dxsize > maxcorr || !isposdef(Hcorr)
        return (xhat, l0, H, false, true)
    else
        return (xcorr, fitfun(xcorr), Matrix(Hcorr), false, false)
    end
end

function estep!(f::EMFit)
    m = f.model
    mus = m.X * f.betas

    inv_sigma = inv(f.sigma)
    logdet_sigma = logdet(f.sigma)
    nfallback = Threads.Atomic{Int}(0)

    Threads.@threads for i = 1:m.nsub
        sub = m.subs[i]

        fitfun = (params) -> gaussianprior(params, view(mus, i, :), inv_sigma, logdet_sigma, view(m.data, m.data.sub .== sub, :), m.likfun)

        x, l, H, fellback, _ = subject_laplace_fit(fitfun, f.x[i, :], inv_sigma)
        fellback && Threads.atomic_add!(nfallback, 1)
        f.x[i, :] = x
        f.l[i] = l
        f.h[:, :, i] = inv(H)
    end
    return (hessian_fallback=nfallback[], skew_fallback=0)
end

# E-step with a third-derivative (skewness) correction to the Laplace
# approximation, applied every iteration. Corrects the classic mode/mean
# mismatch of the Laplace approximation for subjects whose per-subject
# posterior is meaningfully non-Gaussian (e.g. induced by an extreme value
# of a jointly-estimated parameter degrading identifiability of another).
# Cheap relative to the optimization itself: one extra ForwardDiff.jacobian
# of the Hessian per subject per iteration.
function estep_skewcorrect!(f::EMFit; maxcorr=1.0)
    m = f.model
    mus = m.X * f.betas
    inv_sigma = inv(f.sigma)
    logdet_sigma = logdet(f.sigma)
    nfallback_hessian = Threads.Atomic{Int}(0)
    nfallback_skew = Threads.Atomic{Int}(0)

    Threads.@threads for i = 1:m.nsub
        sub = m.subs[i]
        fitfun = (params) -> gaussianprior(params, view(mus, i, :), inv_sigma, logdet_sigma, view(m.data, m.data.sub .== sub, :), m.likfun)

        x, l, H, fellback_h, fellback_s = subject_laplace_fit(fitfun, f.x[i, :], inv_sigma; skewcorrect=true, maxcorr=maxcorr)
        fellback_h && Threads.atomic_add!(nfallback_hessian, 1)
        fellback_s && Threads.atomic_add!(nfallback_skew, 1)
        f.x[i, :] = x
        f.l[i] = l
        f.h[:, :, i] = inv(H)
    end
    return (hessian_fallback=nfallback_hessian[], skew_fallback=nfallback_skew[])
end

function mstep!(f::EMFit)
    # this result from http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf
    m = f.model

    f.betas = inv(m.X' * m.X) * m.X' * f.x
    is_diagonal = (typeof(f.sigma) <: Diagonal)

    proj = I - m.X * inv(m.X' * m.X) * m.X'
    # REML-equivalent dof correction (Helwig notes eq. for Sigma-hat): divide by
    # nsub - nreg rather than nsub, since nreg dof are used up estimating betas
    newsigma = (f.x' * proj * f.x + dropdims(sum(f.h, dims=3), dims=3)) / (m.nsub - m.nreg)

    if is_diagonal
        f.sigma = Diagonal(diag(newsigma))
    else
        f.sigma = newsigma
    end

    nothing
end

#### functions related to error bars

function informationmatrixsigma(sigma, dof)
    # this computes the sub block of the complete information matrix for the sigma parameters
    # it is pretty ugly due to the unrolling of the sigma parameters.
    # the version that calls autograd is much easier to read but gratuitous
    # `dof` is the effective sample size for sigma (nsub - nreg), i.e. the divisor from
    # the REML-corrected sigma
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

            I_ΣΣ[row, col] = (dof / 2) * val
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


function groupinformation(fit::EMFit)
    # full observed (Louis-corrected) information matrix for ALL group-level
    # parameters packed together (betas then sigma, matching packparams) --
    # emcovmtx uses just the beta sub-block of this; ilaplace needs the whole
    # thing since its Laplace penalty is over all group-level parameters
    m = fit.model
    nbetas = prod(size(fit.betas))

    prior = packparams(fit.betas, fit.sigma)

    # the first term is the information matrix of the complete data likelihood
    # (the "complete information")
    # it is block diagonal, with one block for the betas and one for the sigma
    # these are standard MVN formulas

    h1 = zeros(length(prior), length(prior))
    h1[1:nbetas, 1:nbetas] = inv(kron(inv(m.X' * m.X), fit.sigma))
    # nsub - nreg (not nsub) to stay consistent with the REML-corrected sigma in
    # mstep!: sum_i E[r_i r_i'|y] = (nsub-nreg)*sigma at the M-step's fixed point
    # now, not nsub*sigma
    h1[nbetas+1:end, nbetas+1:end] = informationmatrixsigma(fit.sigma, m.nsub - m.nreg)

    # the second term is the missing information
    h2 = missing_information(fit)

    return h1 - h2
end

function emcovmtx(fit::EMFit)
    # compute covariance on the group level model parameters using missing information
    # this version from Tagare "A gentle introduction to the EM algorithm" eq 4.1
    nbetas = prod(size(fit.betas))
    return inv(groupinformation(fit))[1:nbetas, 1:nbetas]
end

"""
    emerrors(fit::EMFit)

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

    # dof from helwig notes: t_{n-p-1}, i.e. nsub - nreg here (nreg already includes
    # the intercept column)
    pvalues = 2 * ccdf.(TDist(m.nsub - m.nreg), abs.(vec(fit.betas')) ./ ses)

    return EMErrors(ses, pvalues, covmtx, fit)
end



#### functions related to model selection 

# aggregate / integrated measures
"""
    lml(fit::EMFit)

Compute the Laplace approximation to the negative integrated log-likelihood of the dataset for a fitted model, i.e. -log p(data | betas, sigma), marginalizing over subject-level parameters via a per-subject Laplace approximation. This is the primary, recommended API.

Follows the same negative-log-likelihood convention as the per-subject `fit.l`: lower values indicate a better fit. `ibic` and `iaic` add their complexity penalties directly to this value.
This marginalizes over the subject-level parameters but note that it is conditional on (not corrected for overfitting due to) the estimated group-level parameters.

---

    lml(x, l, h)

Backward-compatible positional arguments version of `lml`.
"""
function lml(fit::EMFit)
    m = fit.model

    return -m.nparam / 2 * log(2 * pi) * m.nsub + sum(fit.l) - sum(logdet(fit.h[:, :, i]) for i in 1:m.nsub) / 2
end


"""
    ibic(fit::EMFit, ndata=fit.model.nsub)

Compute the Integrated Bayesian Information Criterion (iBIC) for a fitted model: `lml(fit) + k/2*log(ndata)`, where `k` is the number of group-level parameters.

# Arguments
- `fit::EMFit`: The fitted model object returned by `em()`.
- `ndata`: (Optional) The effective sample size for the BIC penalty. Defaults to `fit.model.nsub` -- the number of subjects, which is what actually governs the growth of the group-level Fisher information in this model (see `emcovmtx`/`groupinformation`: the observed information for betas and sigma scales with nsub, not with the number of trials per subject or with nsub*nparam). Pass `ndata` explicitly (e.g. `nsub*ntrials`, as in Huys et al. 2011's original iBIC) for comparability with that convention, or with prior analyses that used it.

---

    ibic(x, l, h, betas, sigma, ndata=size(x,1))

Backward-compatible positional arguments version of `ibic`.
"""
function ibic(fit::EMFit, ndata=fit.model.nsub)
    return lml(fit) + length(packparams(fit.betas, fit.sigma)) / 2 * log(ndata)
end


"""
    iaic(fit::EMFit)

Compute the Integrated Akaike Information Criterion (iAIC) for a fitted model.

---

    iaic(x, l, h, betas, sigma)

Backward-compatible positional arguments version of `iaic`.
"""
function iaic(fit::EMFit)
    return lml(fit) + length(packparams(fit.betas, fit.sigma))
end


"""
    ilaplace(fit::EMFit)

Compute a Laplace approximation to the (negative) integrated log-likelihood of the dataset, correcting for the group-level parameters' own estimation uncertainty using the actual observed information matrix for those parameters, rather than the generic `k/2*log(ndata)` penalty used by `ibic`.

`ibic` and `iaic` approximate the curvature of the group-level marginal likelihood by a generic `ndata^k`-style scaling, which is what the classical BIC/AIC penalties assume in place of the real Fisher information -- avoiding the need to know that curvature exactly, at the cost of needing to choose a value for `ndata` that isn't always obvious in a hierarchical model (see `ibic`). Since `emcovmtx`/`groupinformation` already compute the real observed information matrix for the group-level parameters (needed for `emerrors`'s standard errors anyway), `ilaplace` uses it directly instead of a generic proxy, sidestepping the choice of `ndata` altogether: `lml(fit) + logdet(groupinformation(fit))/2 - k/2*log(2*pi)`, where `k` is the number of group-level parameters.

Follows the same negative-log-likelihood ("loss") convention as `lml`/`ibic`/`iaic`: lower values indicate a better fit. Returns `NaN` (with a warning) if the observed information matrix isn't positive-definite, which would indicate the Laplace approximation itself has broken down for the group-level fit (e.g. too little data to reliably estimate the group-level curvature).

---

    ilaplace(x, l, h, betas, sigma, X)

Backward-compatible positional arguments version of `ilaplace`.
"""
function ilaplace(fit::EMFit)
    k = length(packparams(fit.betas, fit.sigma))
    info = groupinformation(fit)
    if !isposdef(Symmetric(info))
        println("Warning: group-level observed information matrix is not positive-definite; ilaplace is undefined (returning NaN)")
        return NaN
    end
    return lml(fit) + logdet(Symmetric(info)) / 2 - k / 2 * log(2 * pi)
end


# model selection by leave one out cross validation (at the subject level)
# this uses laplace approximation to the marginal likelihood for each subject

"""
    loocv(fit::EMFit; emtol=1e-3, full=false, maxiter=100, skewcorrect=false, skewcorrect_maxcorr=1.0)

Compute per-subject leave-one-subject-out predictive likelihood scores under a fitted model.
Scores are computed from cross-validated group-level parameters, with each subject left out, and using a Laplace approximation to marginalize the subject-level parameters.

# Arguments
- `fit::EMFit`: The fitted model object returned by `em()`.
- `emtol=1e-3`: Stopping point tolerance for relative change in parameters.
- `full=false`: Use a full (vs. diagonal) group-level covariance.
- `maxiter=100`: Maximum EM iterations per-subject.
- `skewcorrect=false`, `skewcorrect_maxcorr=1.0`: Passed through to both the leave-one-out refits (`em()`) and the held-out subject's own likelihood evaluation, so they match whatever `em()` options were used to produce `fit` -- see `em`'s docstring. These aren't inferred from `fit` automatically (nothing on `EMFit` records how it was produced), so pass the same values you used originally if you want consistency between the cross-validated fits and the fit being validated.

---

    loocv(data, subs, startx, X, betas, sigma, likfun; emtol=1e-3, full=false, maxiter=100, skewcorrect=false, skewcorrect_maxcorr=1.0)

Backward-compatible positional arguments version of `loocv`.
"""
function loocv(f::EMFit; emtol=1e-3, full=false, maxiter=100, skewcorrect=false, skewcorrect_maxcorr=1.0)
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
            loo_fit = em(loo_model; startbetas=f.betas, startsigma=f.sigma, emtol=emtol, startx=loostartx, full=full, maxiter=maxiter, quiet=0, skewcorrect=skewcorrect, skewcorrect_maxcorr=skewcorrect_maxcorr)
            newmu = loo_fit.betas' * m.X[i, :]

            liks[i] = heldoutsubject_laplace(newmu, loo_fit.sigma, m.data[m.data[:, :sub].==sub, :], m.likfun; startx=f.x[i, :], skewcorrect=skewcorrect, skewcorrect_maxcorr=skewcorrect_maxcorr)
        catch err
            println(err)
            liks[i] = NaN
        end
    end

    return (liks)
end


function heldoutsubject_laplace(mu, sigma, data, likfun; startx=mu, skewcorrect=false, skewcorrect_maxcorr=1.0)
    nparam = length(mu)

    inv_sigma = inv(sigma)
    logdet_sigma = logdet(sigma)
    fitfun = (x) -> gaussianprior(x, mu, inv_sigma, logdet_sigma, data, likfun)

    x, l, H, _, _ = subject_laplace_fit(fitfun, startx, inv_sigma; skewcorrect=skewcorrect, maxcorr=skewcorrect_maxcorr)

    return -nparam / 2 * log(2 * pi) + l + logdet(H) / 2
end


# attempt to compute the free energy expression as given in Gharamani EM slides

function freeenergy(f::EMFit)
    m = f.model

    mu = m.X * f.betas

    if (det(f.sigma) < 0)
        return NaN
    end

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
        for sub in 1:m.nsub
    )

    # expected LL for the observations
    return val - lml(f)
end
