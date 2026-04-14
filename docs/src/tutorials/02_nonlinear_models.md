# Non-Linear Panel Models in Panelest.jl

```@meta
CurrentModule = Panelest
```

## Setup

```@example nonlinear_models
using Panelest, DataFrames, Random, StatsModels, Vcov, Distributions

Random.seed!(123)
n = 1_000
latent = randn(n)
df = DataFrame(
    id = repeat(1:100, inner=10),
    time = repeat(1:10, 100),
    x1 = randn(n)
)
df.y_counts = rand.(Poisson.(exp.(0.4 .+ 0.3 .* df.x1 .+ 0.2 .* latent)))
df.y_binary = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-0.8 .* df.x1 .- 0.4 .* latent)))
nothing
```

`Panelest.jl` provides specialized tools for non-linear panel models, including binary outcomes and count data.

## Poisson Models

Poisson regressions with fixed effects are highly optimized using a unified IRLS-FE engine.

```@example nonlinear_models
m = fepois(df, @formula(y_counts ~ x1 + fe(id) + fe(time)))
println(m)
```

## Binary Choice Models

### Logit and Probit

You can estimate standard logit and probit models with high-dimensional fixed effects:

```@example nonlinear_models
m_logit = felogit(df, @formula(y_binary ~ x1 + fe(id)))
println(m_logit)
```

Note: In models with binary outcomes and many fixed effects, be mindful of the incidental parameters problem. For logit models, Chamberlain's conditional logit (`clogit`) is often preferred for consistency.

### Chamberlain's Conditional Logit

`clogit` estimates a fixed-effects logit model using the conditional likelihood approach, which remains consistent even as the number of fixed effects grows.

```@example nonlinear_models
# Grouped by :id
m_clogit = clogit(df, @formula(y_binary ~ x1), :id)
println(m_clogit)
```

`Panelest.jl` uses a recursive algorithm (Gail et al., 1981) that efficiently handles groups with any number of treated units.

## Correlated Random Effects (CRE)

For other non-linear models (like probit) where conditional logit is not available, the Correlated Random Effects approach (Wooldridge, 2010) is a robust alternative. It augments the covariates with their group-level means to capture the correlation between the fixed effects and the observed variables.

```@example nonlinear_models
# CRE Probit
m_cre = cre(df, @formula(y_binary ~ x1), :id, family=:probit)
println(m_cre)
```

The `cre` function automatically identifies time-varying variables and computes their group means, adding them to the model before estimation.
