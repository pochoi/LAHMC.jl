module LAHMC

using LinearAlgebra

export LogLikelihood, GaussianLogLikelihood, RLogLikelihood
include("loglikelihood.jl")

export Sampler, LAHMCSampler, HMCSampler
export set_x0!, sample!
include("sampler.jl")

export mcmc!, mcmc
export overrelaxation_mcmc!, overrelaxation

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



#=
function mcmc!(s, N, S::Sampler{T}; burnin = 0, thin = 1) where T
    if length(s) > 0
        set_x0!(S, s[end])
    end
    for k in 1:burnin
        sample!(S)
    end

    append!(s, zeros(T, div(N, thin)))
    s_counter = length(s)
    for k in 1:N
        sample!(S)
        if k % thin == 0
            s_counter += 1
            s[s_counter] = S.x
        end
    end
    return s
end
=#

function overrelaxation_mcmc!(s, N, S::Sampler{T}, index_rank, index_fix; K = 10, subK = 5, burnin = 0, thin = 1) where T
    n = length(s)
    s0 = Vector{T}[s[end]]
    for k in 1:N
        for (i,j) in zip(index_rank, index_fix)
            x1 = overrelaxation(S, s0[end], i, j, K, subK)
            push!(s0, x1)
        end
    end

    append!(s, s0[(burnin+1) .+ (1:thin:N)])

    return s
end

function overrelaxation(S::Sampler{T},
                         x0,
                         index_rank::Integer,
                         index_fix,
                         K::Integer,
                         subK::Integer) where T
    out = Vector{T}[x0]
    for k in 1:K
        set_x0!(S, x0)
        for n in 1:subK
            sample!(S, index_fix)
        end
        push!(out, S.x)
    end

    ii = sortperm(out, lt= (a,b) -> a[index_rank] < b[index_rank])
    r = findall(ii .== 1)[1]
    return out[ii[K-(r-1)+1]]
end








end # module
