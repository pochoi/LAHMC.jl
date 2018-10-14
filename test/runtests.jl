using Test
using LinearAlgebra
using Statistics
using LAHMC

@testset "HMC" begin
    μ = 10*randn(3)
    Σ = [1.0 0.2 -0.2; 0.2 2.0 -0.3 ; -0.2 -0.3 3.0]
    logf = GaussianLogLikelihood(μ, Σ)

    @test typeof(copy(logf)) == typeof(logf)

    pΣL = LowerTriangular([1.0 0.0 0.0; 0.0 sqrt(2.0) 0.0; 0.0 0.0 sqrt(3.0)])
    hmc_sampler0 = HMCSampler(logf, zeros(3), 0.1, 20, pΣL)

    @test typeof(copy(hmc_sampler0)) == typeof(hmc_sampler0)

    N = 20000
    hmc_sample, hmc_sampler = mcmc(zeros(3), N, hmc_sampler0, burnin = 200, thin = 4)
    @test norm(mean(hmc_sample) - μ) < 1.0
    @test norm(cov(hmc_sample) - Σ) < 1.0
end

@testset "LAHMC" begin
    μ = 10*randn(3)
    Σ = [1.0 0.2 -0.2; 0.2 2.0 -0.3 ; -0.2 -0.3 3.0]
    logf = GaussianLogLikelihood(μ, Σ)

    @test typeof(copy(logf)) == typeof(logf)

    pΣL = LowerTriangular([1.0 0.0 0.0; 0.0 sqrt(2.0) 0.0; 0.0 0.0 sqrt(3.0)])
    lahmc_sampler0 = LAHMCSampler(logf, zeros(3), 0.1, 10, 5, pΣL, 0.9)

    @test typeof(copy(lahmc_sampler0)) == typeof(lahmc_sampler0)

    N = 20000
    lahmc_sample, lahmc_sampler = mcmc(zeros(3), N, lahmc_sampler0, burnin = 200, thin = 4)
    @test norm(mean(lahmc_sample) - μ) < 1.0
    @test norm(cov(lahmc_sample) - Σ) < 1.0
end
