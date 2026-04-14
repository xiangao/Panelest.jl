# Getting Started with Panelest.jl

```@meta
CurrentModule = Panelest
```

`Panelest.jl` is a high-performance Julia package for panel data estimation, inspired by the R package `fixest`. It provides efficient tools for linear and non-linear models with high-dimensional fixed effects and flexible clustering.

## Basic Usage

### Linear Fixed Effects (OLS)

To estimate a linear model with multiple fixed effects, use the `feols` function. Fixed effects are specified using the `fe()` term in the formula.

```@example getting_started
using Panelest, DataFrames, StatsModels, Distributions
using Random

Random.seed!(123)

# Generate some data
df = DataFrame(
    id = repeat(1:100, inner=10),
    time = repeat(1:10, 100),
    y = randn(1000),
    x1 = randn(1000),
    y_counts = rand.(Poisson.(exp.(0.2 .+ 0.3 .* randn(1000))))
)

# Estimate OLS with individual and time fixed effects
model = feols(df, @formula(y ~ x1 + fe(id) + fe(time)))
println(model)
```

### Poisson Fixed Effects

Poisson models are estimated using `fepois`. These are particularly fast in `Panelest.jl` due to the integrated IRLS-FE algorithm.

```@example getting_started
# Estimate Poisson with fixed effects
model_pois = fepois(df, @formula(y_counts ~ x1 + fe(id)))
println(model_pois)
```

## Clustered Standard Errors

`Panelest.jl` supports one-way and multi-way clustering through `Vcov.jl`.

```@example getting_started
using Vcov

# One-way clustering on 'id'
model = feols(df, @formula(y ~ x1 + fe(id)), vcov = Vcov.cluster(:id))
println(model)
```

## Non-linear Models

Beyond OLS and Poisson, `Panelest.jl` supports:

*   **Logit/Probit**: `felogit` and `feprobit`.
*   **Conditional Logit**: `clogit(df, @formula(y ~ x1), :id)` for Chamberlain's conditional logit.
*   **Correlated Random Effects (CRE)**: `cre(df, @formula(y ~ x1), :id, family=:probit)` for Wooldridge's CRE approach.

## Performance

`Panelest.jl` matches the performance of R's `fixest` by absorbing fixed effects directly within each iteration of the IRLS loop for GLMs. For OLS, it leverages the high-performance solvers in `FixedEffects.jl`.
