module LAHMC

using LinearAlgebra

export LogLikelihood, GaussianLogLikelihood
include("loglikelihood.jl")

export Sampler, LAHMCSampler, HMCSampler
export set_x0!, sample!
include("sampler.jl")

export mcmc!, mcmc

function mcmc!(s, N, S::Sampler{T}; burnin = 0, thin = 1) where T
    if length(s) > 0
        set_x0!(S, s[end])
    end
    for k in 1:burnin
        sample!(S)
    end
    for k in 1:N
        sample!(S)
        if k % thin == 0
            push!(s, S.x)
        end
    end
    return s
end

function mcmc(x0, N, S0::Sampler{T}; kwargs...) where T
    S = copy(S0)
    set_x0!(S, x0)
    s = Array{T, 1}[]
    mcmc!(s, N, S; kwargs...)
    return s, S
end

end # module
