abstract type LogLikelihood{T} end

struct GaussianLogLikelihood{T, V<:AbstractVector{T}, S<:AbstractMatrix{T}} <: LogLikelihood{T}
    μ::V
    Σ::S
end

Base.copy(logf::GaussianLogLikelihood) = GaussianLogLikelihood(copy(logf.μ), copy(logf.Σ))

function (logf::GaussianLogLikelihood{T})(x::V) where {T, V<:AbstractVector{T}}
    T(-0.5)*dot(x-logf.μ, logf.Σ\(x-logf.μ))
end
(logf::GaussianLogLikelihood)(x, ::Val{:v}) = logf(x)
(logf::GaussianLogLikelihood)(x, ::Val{:g}) = -logf.Σ\(x - logf.μ)
function (logf::GaussianLogLikelihood{T})(x::V, ::Val{:vg}) where {T, V<:AbstractVector{T}}
    invΣx = logf.Σ\(x - logf.μ)
    return T(-0.5)*dot(x - logf.μ, invΣx), -invΣx
end


struct RLogLikelihood{T, V <: LogLikelihood{T}, W <: AbstractMatrix{T}} <: LogLikelihood{T}
    orglogf::V
    R::W
end

Base.copy(logf::RLogLikelihood) = RLogLikelihood(copy(logf.orglogf), copy(logf.R))

function (logf::RLogLikelihood{T})(x::V) where {T, V<:AbstractVector{T}}
    rx = logf.R*x
    return logf.orglogf(rx)
end

(logf::RLogLikelihood)(x, ::Val{:v}) = logf(x)

function (logf::RLogLikelihood{T})(x::V, ::Val{:g}) where {T, V<:AbstractVector{T}}
    rx = logf.R*x
    return R'*logf.orglogf(rx, Val(:g))
end

function (logf::RLogLikelihood{T})(x::V, ::Val{:vg}) where {T, V<:AbstractVector{T}}
    rx = logf.R*x
    v, g = logf.orglogf(rx, Val(:vg))
    return v, logf.R'*g
end



