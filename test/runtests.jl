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

    @testset "Instrumental Variables" begin
        Random.seed!(123)
        n_iv = 5000

        # DGP: y = 1.0*x_endo + 0.5*x_exo + eps
        #       x_endo = 0.8*z1 + 0.5*eps + nu  (endogenous: correlated with eps)
        z1 = randn(n_iv)
        z2 = randn(n_iv)
        eps_iv = randn(n_iv)
        nu = randn(n_iv)
        x_endo = 0.8 .* z1 .+ 0.5 .* eps_iv .+ nu
        x_exo = randn(n_iv)
        y_iv = 1.0 .* x_endo .+ 0.5 .* x_exo .+ eps_iv

        # Group FE for FE tests
        gid = repeat(1:50, inner=100)
        gfe = randn(50)
        y_iv_fe = y_iv .+ gfe[gid]

        df_iv = DataFrame(
            y = y_iv, y_fe = y_iv_fe, x_endo = x_endo, x_exo = x_exo,
            z1 = z1, z2 = z2, gid = gid
        )

        @testset "Basic IV (just-identified)" begin
            m = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z1)
            @test isapprox(coef(m)[1], 1.0, atol=0.15)   # endo coef
            # coef[2] is intercept, coef[3] is x_exo
            @test isapprox(coef(m)[3], 0.5, atol=0.15)   # exo coef
            @test nobs(m) == n_iv
            @test length(stderror(m)) == 3  # endo + intercept + x_exo
            @test m.diagnostics.sargan === nothing  # just-identified
        end

        @testset "IV with FE" begin
            m_fe = feiv(df_iv, @formula(y_fe ~ x_exo + fe(gid)), endo=:x_endo, inst=:z1)
            @test isapprox(coef(m_fe)[1], 1.0, atol=0.15)
            @test isapprox(coef(m_fe)[2], 0.5, atol=0.15)
        end

        @testset "Overidentified (Sargan test)" begin
            # z2 is a valid (but irrelevant) instrument — add noise correlation with x_endo
            df_iv.x_endo2 = 0.8 .* z1 .+ 0.3 .* z2 .+ 0.5 .* eps_iv .+ nu
            df_iv.y2 = 1.0 .* df_iv.x_endo2 .+ 0.5 .* x_exo .+ eps_iv
            m_over = feiv(df_iv, @formula(y2 ~ x_exo), endo=:x_endo2, inst=[:z1, :z2])
            @test m_over.diagnostics.sargan !== nothing
            @test m_over.diagnostics.sargan.df == 1.0  # 2 inst - 1 endo
            @test m_over.diagnostics.sargan.p > 0.01   # valid instruments → fail to reject
        end

        @testset "OLS vs IV bias" begin
            # OLS should be biased (x_endo correlated with eps)
            m_ols_iv = feols(df_iv, @formula(y ~ x_endo + x_exo))
            m_iv = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z1)
            # OLS endo coef should be biased upward (positive correlation eps→x_endo)
            ols_bias = abs(coef(m_ols_iv)[1] - 1.0)
            iv_bias = abs(coef(m_iv)[1] - 1.0)
            @test ols_bias > iv_bias  # IV should be less biased
        end

        @testset "First-stage F-stat" begin
            m = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z1)
            @test m.diagnostics.first_stage_F[1] > 10.0  # strong instrument

            # Weak instrument test: pure noise, no correlation with x_endo
            Random.seed!(999)
            df_iv.z_weak = randn(n_iv)
            m_weak = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z_weak)
            @test m_weak.diagnostics.first_stage_F[1] < 10.0  # weak instrument
        end

        @testset "Multiple endogenous variables" begin
            Random.seed!(456)
            z_a = randn(n_iv)
            z_b = randn(n_iv)
            eps2 = randn(n_iv)
            endo_a = 0.7 .* z_a .+ 0.3 .* eps2 .+ randn(n_iv)
            endo_b = 0.6 .* z_b .+ 0.2 .* eps2 .+ randn(n_iv)
            y_multi = 1.0 .* endo_a .+ (-0.5) .* endo_b .+ eps2

            df_multi = DataFrame(y=y_multi, endo_a=endo_a, endo_b=endo_b,
                                 z_a=z_a, z_b=z_b)
            m_multi = feiv(df_multi, @formula(y ~ 0), endo=[:endo_a, :endo_b], inst=[:z_a, :z_b])
            @test isapprox(coef(m_multi)[1], 1.0, atol=0.2)
            @test isapprox(coef(m_multi)[2], -0.5, atol=0.2)
        end

        @testset "Clustered SEs" begin
            m_simple = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z1)
            m_cluster = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z1,
                            vcov_type=Vcov.cluster(:gid))
            # Clustered SEs should generally differ from simple SEs
            @test stderror(m_simple) != stderror(m_cluster)
        end

        @testset "show method" begin
            m = feiv(df_iv, @formula(y ~ x_exo), endo=:x_endo, inst=:z1)
            buf = IOBuffer()
            show(buf, m)
            output = String(take!(buf))
            @test occursin("IV/2SLS", output)
            @test occursin("x_endo", output)
            @test occursin("First-stage F", output)
            @test occursin("Wu-Hausman", output)
        end
    end
end
