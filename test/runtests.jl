using Test
using Panelest
using DataFrames
using Random
using Distributions
using StatsModels
using Vcov
using StatsAPI
using StatsBase
using StatsFuns

@testset "Panelest.jl" begin
    Random.seed!(42)
    N = 10000
    K = 50
    df = DataFrame(
        id = repeat(1:K, inner = Int(N/K)),
        x1 = randn(N),
        x2 = randn(N)
    )

    id_fe = randn(K)
    eta = 0.5 .* df.x1 .- 0.3 .* df.x2 .+ id_fe[df.id]
    
    df.y_ols = eta .+ randn(N)
    df.y_pois = [rand(Poisson(exp(e))) for e in eta]
    df.y_probit = [rand(Bernoulli(normcdf(e - mean(eta)))) for e in eta]
    
    df.y_clogit = zeros(Int, N)
    for g in groupby(df, :id)
        g.y_clogit[rand(1:nrow(g))] = 1
    end
    
    @testset "Linear Models" begin
        m_ols = feols(df, @formula(y_ols ~ x1 + x2 + fe(id)))
        @test m_ols.converged
        @test isapprox(coef(m_ols)[1], 0.5, atol=0.1)
        @test isapprox(coef(m_ols)[2], -0.3, atol=0.1)
    end
    
    @testset "Nonlinear Models" begin
        m_pois = fepois(df, @formula(y_pois ~ x1 + x2 + fe(id)))
        @test m_pois.converged
        @test isapprox(coef(m_pois)[1], 0.5, atol=0.1)
        
        m_probit = feprobit(df, @formula(y_probit ~ x1 + x2 + fe(id)))
        @test m_probit.converged
        @test isapprox(coef(m_probit)[1], 0.5, atol=0.2)
    end
    
    @testset "Conditional Logit & CRE" begin
        m_clogit = clogit(df, @formula(y_clogit ~ x1 + x2), :id)
        @test m_clogit.converged
        
        m_cre = cre(df, @formula(y_probit ~ x1 + x2), :id, family = :probit)
        @test m_cre.converged
    end
end
