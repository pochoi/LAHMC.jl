module LAHMC

using LinearAlgebra

export LogLikelihood, GaussianLogLikelihood
include("loglikelihood.jl")

export Sampler, LAHMCSampler, HMCSampler
export set_x0!, sample!
include("sampler.jl")

export mcmc!, mcmc
export overrelaxation_mcmc!

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

function overrelaxation_mcmc!(s, N, S::Sampler{T}; K = 10, subK = 5, burnin = 0, thin = 1) where T
    if length(s) > 0
        set_x0!(S, s[end])
    end

    for k in 1:N
        x0 = s[end]
        x1 = overrelaxation(S, x0, 1, K, subK)
        x2 = overrelaxation(S, x1, lmax+2, K, subK)
        x3 = overrelaxation(S, x2, 2, K, subK)
        x4 = overrelaxation(S, x3, lmax+3, K, subK)

        push!(s, x1)
        push!(s, x2)
        push!(s, x3)
        push!(s, x4)
    end

    return s
end

function overrelaxation(S::Sampler{T},
                         x0::T,
                         index::Integer,
                         K::Integer,
                         subK::Integer) where T
    out = T[x0]
    for k in 1:K
        set_x0!(S, x0)
        for n in 1:subK
            sample!(S)
        end
        push!(out, S.x)
    end

    ii = sortperm(out, lt= (a,b) -> a[index] < b[index])
    r = findall(ii .== 1)
    return out[ii[K-r+1]]
end

function rank(out)
    x0 = out[1]
    sorted_out = sort(out)
    index = searchsorted(sorted_out, x0)

    if index.start == index.stop
        r = index.start - 1
    else
        r = (index.start - 1) + _sample(ones(length(index))) - 1
    end
    return r, sorted_out
end


function mcmc(x0, N, S0::Sampler{T}; kwargs...) where T
    S = copy(S0)
    set_x0!(S, x0)
    s = Array{T, 1}[]
    mcmc!(s, N, S; kwargs...)
    return s, S
end







end # module
