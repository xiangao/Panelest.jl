# Panelest.jl

`Panelest.jl` is a high-performance Julia package for panel data estimation. It is designed to match the speed and user-friendliness of R's `fixest` while leveraging Julia's native performance for non-linear models.

## Features

- **High-Performance IRLS-FE**: Efficiently absorbs fixed effects within the iteratively reweighted least squares loop for Poisson, Logit, and Probit models.
- **Multiple Fixed Effects**: Support for high-dimensional individual and time fixed effects using `FixedEffects.jl`.
- **Flexible Clustering**: Robust and multi-way clustered standard errors via `Vcov.jl`.
- **Non-Linear Panel Methods**: Includes Chamberlain's Conditional Logit (recursive algorithm) and Wooldridge's Correlated Random Effects (CRE).

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
