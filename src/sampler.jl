abstract type Sampler{T} end

function set_x0!(S::Sampler{T}, x0::AbstractVector{T}) where T
    S.x = x0
    return S
end

mutable struct HMCSampler{T,
                          LL <: LogLikelihood{T},
                          V<:AbstractVector{T},
                          S<:AbstractMatrix{T},
                          SS<:AbstractMatrix{T}
                          } <: Sampler{T}
    logf::LL
    x::V
    ϵ::T
    M::Int
    ΣL::S
    Σ2::SS
end

function HMCSampler(logf::LogLikelihood{T}, x::AbstractVector{T}, ϵ, M, ΣL) where T
    Σ2 = ΣL * ΣL'
    HMCSampler{eltype(x), typeof(logf), typeof(x), typeof(ΣL), typeof(Σ2)}(logf, x, ϵ, M, ΣL, Σ2)
end

function Base.copy(S::HMCSampler)
    HMCSampler(copy(S.logf), copy(S.x), copy(S.ϵ), copy(S.M), copy(S.ΣL), copy(S.Σ2))
end



mutable struct LAHMCSampler{T,
                            LL<:LogLikelihood{T},
                            V<:AbstractVector{T},
                            S<:AbstractMatrix{T},
                            SS<:AbstractMatrix{T}} <: Sampler{T}
    logf::LL
    x::V
    ϵ::T
    M::Int
    K::Int
    ΣL::S
    Σ2::SS
    β::Float64
    p::V
    k::Vector{Int}
end

function LAHMCSampler(logf::LogLikelihood{T}, x::AbstractVector{T}, ϵ, M, K, ΣL, β) where T
    p = ΣL*randn(T, length(x))
    Σ2 = ΣL * ΣL'
    LAHMCSampler{eltype(x), typeof(logf), typeof(x), typeof(ΣL), typeof(Σ2)}(logf, x, ϵ, M, K, ΣL, Σ2, β, p, zeros(Int, K+1))
end

function Base.copy(S::LAHMCSampler)
    LAHMCSampler(copy(S.logf), copy(S.x), copy(S.ϵ), S.M, S.K, copy(S.ΣL), copy(S.Σ2), S.β, copy(S.p), copy(S.k))
end

function momentum_resample!(S::LAHMCSampler{T}; β = S.β) where T
    pp1 = (S.ΣL * randn(T, length(S.x))) .* sqrt(β)
    pp2 = S.p .* sqrt(1 - β)
    S.p = pp1 + pp2
    return S
end

function sample!(S::HMCSampler{T}) where T
    ΣL = S.ΣL
    Σ2 = S.Σ2

    ϵ = S.ϵ
    M = S.M
    logf = S.logf

    x0 = S.x
    p0 = ΣL * randn(T, length(x0))

    x1, p1, logf1, grad1, logf0, grad0 = leapfrog(x0, p0, ϵ, M, Σ2, logf)
    p1 = -p1

    current_U  = -logf0
    current_K  = KEngry(ΣL, p0)
    proposed_U = -logf1
    proposed_K = KEngry(ΣL, p1)

    prob = exp(current_U - proposed_U + current_K - proposed_K)
    if rand() < prob
        S.x = x1
    end

    return S
end

KEngry(ΣL, p) = (half = ΣL\p; 0.5*dot(half,half))
∇KEngry(Σ2, p) = Σ2\p

function leapfrog(x0::AbstractVector{T}, p0::AbstractVector{T}, ϵ, M, Σ2, logf::LogLikelihood{T}) where T
    logf0, grad0 = logf(x0, Val(:vg))
    x1 = copy(x0)
    p1 = copy(p0)
    p1 += T(0.5) * ϵ * grad0
    logf1 = T(0.0)
    grad1 = similar(grad0)
    for i in 1:M
        x1          += ϵ * ∇KEngry(Σ2, p1)
        logf1, grad1 = logf(x1, Val(:vg))
        p1          += ϵ * grad1
    end
    p1 -= T(0.5) * ϵ * grad1
    return x1, p1, logf1, grad1, logf0, grad0
end

function sample!(S::LAHMCSampler{T}) where T
    ΣL = S.ΣL
    Σ2 = ΣL * ΣL'

    x0 = S.x[:]
    x1 = S.x[:]

    p0 = S.p[:]
    p1 = S.p[:]

    K = S.K
    M = S.M
    ϵ = S.ϵ
    β = S.β
    logf = S.logf

    x_candidate = [x0 for k in 1:K+1]
    p_candidate = [p0 for k in 1:K+1]
    rand_comparison = rand()
    p_cum = 0.0
    C = zeros(Float64, K+1, K+1)
    fill!(C, NaN)

    for k in 1:K
        x_new, p_new= leapfrog( x_candidate[k], p_candidate[k],
                                ϵ, M, Σ2, logf)

        x_candidate[k+1] = x_new
        p_candidate[k+1] = p_new
        p_cum, Cl = leap_prob_recurse( x_candidate[1:(k+1)], p_candidate[1:(k+1)],
                                       C[1:(k+1),1:(k+1)], ΣL, logf)
        C[1:(k+1),1:(k+1)] = Cl

        if p_cum >= rand_comparison
            S.k[k] += 1
            x1 = x_new
            p1 = p_new
            break
        end
    end
    if p_cum < rand_comparison
        S.k[K+1] += 1
        x1 = copy(x0)
        p1 = -p0
    end

    pp1 = (ΣL * randn(T, length(x0))) .* sqrt(β)
    pp2 = p1 .* sqrt(1 - β)
    S.p = pp1 + pp2

    S.x = x1
    return S
end

function leap_prob(θ₁::AbstractVector{T}, r₁::AbstractVector{T},
                   θ₂::AbstractVector{T}, r₂::AbstractVector{T},
                   ΣL, logfgrad::LogLikelihood{T}) where T
    logf1 = logfgrad(θ₁, Val(:v))
    logf2 = logfgrad(θ₂, Val(:v))

    Ediff = (-logf1 + KEngry(ΣL, r₁))-(-logf2 + KEngry(ΣL, r₂))
    return min(1.0, exp(Ediff))
end

function leap_prob_recurse(θ_chain, r_chain, C, ΣL, logfgrad::LogLikelihood{T}) where T
    if isfinite(C[1,end])
        # we've already visited this leaf
        cumu = C[1,end]
        return cumu, C
    end

    if length(θ_chain) == 2
        p_acc = leap_prob(θ_chain[1], r_chain[1], θ_chain[2], r_chain[2], ΣL, logfgrad)
        C[1,end] = p_acc
        return p_acc, C
    end

    cum_forward, Cl = leap_prob_recurse(θ_chain[1:end-1],
                                        r_chain[1:end-1],
                                        C[1:end-1, 1:end-1], ΣL, logfgrad)
    C[1:end-1,1:end-1] = Cl
    cum_reverse, Cl = leap_prob_recurse(θ_chain[end:-1:2],
                                        r_chain[end:-1:2],
                                        C[end:-1:2, end:-1:2], ΣL, logfgrad)
    C[end:-1:2, end:-1:2] = Cl

    _H0 = logfgrad(θ_chain[1], Val(:v))
    _H1 = logfgrad(θ_chain[end], Val(:v))

    H0 = -_H0 + KEngry(ΣL, r_chain[1])
    H1 = -_H1 + KEngry(ΣL, r_chain[end])
    Ediff = H0 - H1

    start_state_ratio = exp(Ediff)

    if isnan(start_state_ratio*(1 - cum_reverse))
        prob = 1 - cum_forward
    else
        prob = min(1 - cum_forward, start_state_ratio*(1 - cum_reverse))
    end

    cumu = cum_forward + prob
    C[1, end] = cumu

    return cumu, C
end
