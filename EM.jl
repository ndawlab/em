module EM

using DataFrames
using ForwardDiff
using Optim
using LinearAlgebra       # for tr, diagonal
using StatsFuns           # logsumexp
using SpecialFunctions    # for erf
using Statistics          # for mean
using Distributions		  # for tDist

export em,emerrors,lml,ibic,iaic,loocv,qlik,jianlik,seqlik,simq,simseq,simjian

include("emcore.jl")
include("emutils.jl")
include("emlikfuns.jl")

end