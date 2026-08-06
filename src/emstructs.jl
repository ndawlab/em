using Printf
using LinearAlgebra
using DataFrames

"""
    EMModel(data::DataFrame, subs, X, nparam, likfun; reg_names=nothing, param_names=nothing)
    EMModel(data::DataFrame, subs, X, likfun; reg_names=nothing, param_names=nothing)

Structure to hold inputs and dimensions needed to fit a hierarchical model using Expectation-Maximization (EM).

# Fields
- `data::DataFrame`: A data frame containing the data to be fit. Should have a column `sub` indicating the subject for each observation.
- `subs`: A vector or range of subject IDs.
- `X::Matrix{Float64}`: The design matrix, with a row per subject and a column per group-level predictor.
- `nparam::Int`: The number of subject-level parameters. If omitted, defaults to 0 and will be inferred during fitting.
- `nsub::Int`: The number of subjects (computed automatically).
- `nreg::Int`: The number of regressors (computed automatically).
- `likfun::Function`: The subject-level likelihood function.
- `reg_names::Union{Vector{String}, Nothing}`: Optional vector of custom regressor names for output formatting.
- `param_names::Union{Vector{String}, Nothing}`: Optional vector of custom parameter names for output formatting.
"""
struct EMModel
    data::DataFrame
    subs::AbstractVector{<:Any}
    X::Matrix{Float64}
    nparam::Int
    nsub::Int
    nreg::Int
    likfun::Function
    reg_names::Union{Vector{String}, Nothing}
    param_names::Union{Vector{String}, Nothing}
end

# Outer constructors for EMModel supporting vector/matrix X and keyword-based reg_names/param_names
function EMModel(data::DataFrame, subs::AbstractVector, X::AbstractVecOrMat, nparam::Int, likfun::Function; reg_names=nothing, param_names=nothing)
    X_mat = X isa AbstractVector ? reshape(Float64.(X), :, 1) : Matrix{Float64}(X)
    return EMModel(data, subs, X_mat, nparam, length(subs), size(X_mat, 2), likfun, reg_names, param_names)
end

function EMModel(data::DataFrame, subs::AbstractVector, X::AbstractVecOrMat, likfun::Function; reg_names=nothing, param_names=nothing)
    return EMModel(data, subs, X, 0, likfun; reg_names=reg_names, param_names=param_names)
end

"""
    EMFit(betas, sigma, x, l, h, model)

Represents the fitted results and current state of the Expectation-Maximization (EM) model estimation.

# Fields
- `betas::Matrix{Float64}`: The estimated group-level coefficients.
- `sigma::Union{Diagonal, Matrix}`: The estimated group-level covariance matrix or vector of variances.
- `x::Matrix{Float64}`: The per-subject parameters (subjects x parameters).
- `l::Vector{Float64}`: The per-subject negative log-likelihoods.
- `h::Array{Float64, 3}`: The per-subject inverse Hessians.
- `model::EMModel`: Reference to the model definition.

Supports 5-element iteration protocol for backward compatibility destructuring:
```julia
betas, sigma, x, l, h = fit
```
"""
mutable struct EMFit
    betas::Matrix{Float64}
    sigma::Union{Diagonal{Float64, Vector{Float64}}, Matrix{Float64}}
    x::Matrix{Float64}
    l::Vector{Float64}
    h::Array{Float64, 3}
    model::EMModel
end

# Implement iteration protocol for EMFit (destructuring into 5-tuple)
Base.iterate(fit::EMFit) = (fit.betas, 1)
function Base.iterate(fit::EMFit, state)
    if state == 1
        return (fit.sigma, 2)
    elseif state == 2
        return (fit.x, 3)
    elseif state == 3
        return (fit.l, 4)
    elseif state == 4
        return (fit.h, 5)
    else
        return nothing
    end
end

function Base.show(io::IO, mime::MIME"text/plain", fit::EMFit)
    nreg, nparam = size(fit.betas)
    nsub = fit.model.nsub
    
    println(io, "EM model estimation results:")
    println(io, "--------------------------------------------------------")
    println(io, "Subjects: ", nsub)
    println(io, "Regressors: ", nreg)
    println(io, "Parameters: ", nparam)
    println(io, "--------------------------------------------------------")
    
    # Process regressor and parameter names
    reg_names = fit.model.reg_names !== nothing ? copy(fit.model.reg_names) : ["Reg $i" for i in 1:nreg]
    param_names = fit.model.param_names !== nothing ? copy(fit.model.param_names) : ["Param $j" for j in 1:nparam]
    if length(reg_names) < nreg
        append!(reg_names, ["Reg $i" for i in (length(reg_names)+1):nreg])
    end
    if length(param_names) < nparam
        append!(param_names, ["Param $j" for j in (length(param_names)+1):nparam])
    end
    
    function truncate_str(s, n)
        length(s) > n ? s[1:n-3] * "..." : s
    end

    function print_limited_matrix(mat::AbstractMatrix, row_names::Vector{String}, col_names::Vector{String}; max_rows=6, max_cols=6)
        r, c = size(mat)
        
        # Print headers
        show_c = min(c, max_cols)
        @printf(io, "%-12s", "")
        for j in 1:show_c
            @printf(io, " %-11s", truncate_str(col_names[j], 11))
        end
        if c > show_c
            @printf(io, " %-11s", "...")
        end
        println(io)
        
        # Print rows
        show_r = min(r, max_rows)
        for i in 1:show_r
            @printf(io, "%-12s", truncate_str(row_names[i], 11) * ":")
            for j in 1:show_c
                @printf(io, " %-11.4f", mat[i, j])
            end
            if c > show_c
                @printf(io, " %-11s", "...")
            end
            println(io)
        end
        if r > show_r
            @printf(io, "%-12s", "...")
            for j in 1:show_c
                @printf(io, " %-11s", "...")
            end
            if c > show_c
                @printf(io, " %-11s", "...")
            end
            println(io)
        end
    end

    println(io, "Group-level Coefficients (betas):")
    print_limited_matrix(fit.betas, reg_names, param_names)
    
    println(io)
    println(io, "Group-level Covariance (sigma):")
    print_limited_matrix(fit.sigma, param_names, param_names)
    
    println(io)
    println(io, "Subject-level results:")
    println(io, "  x: ", nsub, " x ", nparam, " subject-level parameters")
    @printf(io, "  l: %d subject-level negative log-likelihoods (sum: %.4f)\n", nsub, sum(fit.l))
    println(io, "  h: ", nsub, " x ", nparam, " x ", nparam, " subject-level inverse Hessians")
    println(io, "--------------------------------------------------------")
end

"""
    EMErrors(ses, pvalues, covmtx, fit)

Represents post-estimation standard errors, p-values, and covariances of the group-level parameters.

# Fields
- `ses::Vector{Float64}`: Standard errors per coefficient.
- `pvalues::Vector{Float64}`: p-values for the null hypothesis that each coefficient = 0.
- `covmtx::Matrix{Float64}`: The covariance matrix over the coefficients.
- `fit::EMFit`: Reference to the parent model fit object.

Supports 3-element iteration protocol for backward compatibility destructuring:
```julia
ses, pvals, covmtx = errors
```
Provides formatted regression table printing via `Base.show`.
"""
struct EMErrors
    ses::Vector{Float64}
    pvalues::Vector{Float64}
    covmtx::Matrix{Float64}
    fit::EMFit
end

# Implement iteration protocol for EMErrors (destructuring into 3-tuple)
Base.iterate(errs::EMErrors) = (errs.ses, 1)
function Base.iterate(errs::EMErrors, state)
    if state == 1
        return (errs.pvalues, 2)
    elseif state == 2
        return (errs.covmtx, 3)
    else
        return nothing
    end
end

# Formatted significance table printing for EMErrors
function Base.show(io::IO, mime::MIME"text/plain", errs::EMErrors)
    fit = errs.fit
    betas = fit.betas
    nreg, nparam = size(betas)
    
    # Set default regressor and parameter names if not provided
    reg_names = errs.fit.model.reg_names !== nothing ? copy(errs.fit.model.reg_names) : ["Reg $i" for i in 1:nreg]
    param_names = errs.fit.model.param_names !== nothing ? copy(errs.fit.model.param_names) : ["Param $j" for j in 1:nparam]
    
    # Ensure they are of correct lengths, if they were custom-supplied but wrong size
    if length(reg_names) < nreg
        append!(reg_names, ["Reg $i" for i in (length(reg_names)+1):nreg])
    end
    if length(param_names) < nparam
        append!(param_names, ["Param $j" for j in (length(param_names)+1):nparam])
    end
    
    println(io, "========================================================================")
    @printf(io, "%-16s %-16s %10s %12s %10s    %-7s\n", "Regressor", "Parameter", "Estimate", "Std.Error", "t-value", "p-value")
    println(io, "========================================================================")
    
    for r in 1:nreg
        for p in 1:nparam
            idx = (r-1)*nparam + p
            est = betas[r, p]
            se = errs.ses[idx]
            pval = errs.pvalues[idx]
            
            tval = se > 0 ? est / se : NaN
            
            # Print p-value format
            p_str = pval < 0.0001 ? "<0.0001" : @sprintf("%.4f", pval)
            
            # Get stars
            stars = if pval < 0.001
                "***"
            elseif pval < 0.01
                "**"
            elseif pval < 0.05
                "*"
            elseif pval < 0.1
                "."
            else
                " "
            end
            
            @printf(io, "%-16s %-16s %10.4f %12.4f %10.2f    %-7s %s\n", 
                    reg_names[r], param_names[p], est, se, tval, p_str, stars)
        end
    end
    println(io, "========================================================================")
    print(io, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
end
