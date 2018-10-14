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
