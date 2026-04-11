# Getting Started with Panelest.jl

`Panelest.jl` is a high-performance Julia package for panel data estimation, inspired by the R package `fixest`. It provides efficient tools for linear and non-linear models with high-dimensional fixed effects and flexible clustering.

## Basic Usage

### Linear Fixed Effects (OLS)

To estimate a linear model with multiple fixed effects, use the `feols` function. Fixed effects are specified using the `fe()` term in the formula.

```julia
using Panelest, DataFrames, StatsModels

# Generate some data
df = DataFrame(
    id = repeat(1:100, inner=10),
    time = repeat(1:10, 100),
    y = randn(1000),
    x1 = randn(1000)
)

# Estimate OLS with individual and time fixed effects
model = feols(df, @formula(y ~ x1 + fe(id) + fe(time)))
println(model)
```

**Output:**
```text
Panelest Model: :ols
Number of obs: 1000
Converged: true
Iterations: 1
────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
x1            -0.049562    0.032896    -1.51    0.1319   -0.114036   0.014912
────────────────────────────────────────────────────────────────────────
```

### Poisson Fixed Effects

Poisson models are estimated using `fepois`. These are particularly fast in `Panelest.jl` due to the integrated IRLS-FE algorithm.

```julia
# Estimate Poisson with fixed effects
model_pois = fepois(df, @formula(y_counts ~ x1 + fe(id)))
println(model_pois)
```

**Output:**
```text
Panelest Model: :poisson
Number of obs: 1000
Converged: true
Iterations: 5
────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
x1             3.006188    0.031403    95.73    <1e-99   2.944638   3.067737
────────────────────────────────────────────────────────────────────────
```

## Clustered Standard Errors

`Panelest.jl` supports one-way and multi-way clustering through `Vcov.jl`.

```julia
using Vcov

# One-way clustering on 'id'
model = feols(df, @formula(y ~ x1 + fe(id)), vcov = Vcov.cluster(:id))
println(model)
```

**Output:**
```text
Panelest Model: :ols
Number of obs: 1000
Converged: true
Iterations: 1
────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
x1            -0.049679    0.033075    -1.50    0.1331   -0.114503   0.015146
────────────────────────────────────────────────────────────────────────
```

## Non-linear Models

Beyond OLS and Poisson, `Panelest.jl` supports:

*   **Logit/Probit**: `felogit` and `feprobit`.
*   **Conditional Logit**: `clogit(df, @formula(y ~ x1), :id)` for Chamberlain's conditional logit.
*   **Correlated Random Effects (CRE)**: `cre(df, @formula(y ~ x1), :id, family=:probit)` for Wooldridge's CRE approach.

## Performance

`Panelest.jl` matches the performance of R's `fixest` by absorbing fixed effects directly within each iteration of the IRLS loop for GLMs. For OLS, it leverages the high-performance solvers in `FixedEffects.jl`.
