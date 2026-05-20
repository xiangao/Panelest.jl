# Panelest.jl

`Panelest.jl` estimates panel models with high-dimensional fixed effects. I use
it as a Julia version of the workflows I usually run with `fixest`: OLS,
Poisson, logit/probit, IV, clustering, and formulas with absorbed fixed effects.

## What is included

- IRLS with fixed effects for Poisson, logit, and probit models.
- Multiple fixed effects through `FixedEffects.jl`.
- One-way and multi-way clustered standard errors through `Vcov.jl`.
- Chamberlain conditional logit and Wooldridge correlated random effects.

## Installation

```julia
using Pkg
Pkg.add("Panelest")
```

## Quick Start

```julia
using Panelest, DataFrames, StatsModels, Vcov

# OLS with fixed effects and clustered SE
model = feols(df, @formula(y ~ x1 + fe(id) + fe(time)), vcov = Vcov.cluster(:id))

# Poisson with fixed effects
model_pois = fepois(df, @formula(y_count ~ x1 + fe(id)))
```
