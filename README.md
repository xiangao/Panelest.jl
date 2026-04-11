# Panelest.jl

A high-performance Julia package for estimating panel data models with high-dimensional fixed effects. Fast and memory-efficient, with support for out-of-core processing via DuckDB.

## Models

| Function   | Model                        |
|------------|------------------------------|
| `feols`    | OLS                          |
| `feiv`     | Instrumental Variables / 2SLS|
| `fepois`   | Poisson                      |
| `felogit`  | Logit                        |
| `feprobit` | Probit                       |
| `clogit`   | Conditional Logit            |
| `cre`      | Correlated Random Effects    |

## Performance

Panelest.jl matches or beats [fixest](https://lrberge.github.io/fixest/) (R/C++) on OLS:

| Sample Size | Fixed Effects    | fixest (R) | Panelest.jl | Ratio        |
|-------------|------------------|------------|-------------|--------------|
| 100K        | 2 FE (simple)    | 0.031s     | 0.024s      | **0.76x** ✅ |
| 1M          | 2 FE (simple)    | 0.618s     | 0.45s       | **0.73x** ✅ |
| 1M          | 3 FE (simple)    | 0.527s     | 0.45s       | **0.85x** ✅ |
| 1M          | 2 FE (difficult) | 6.17s      | 4.97s       | **0.81x** ✅ |

OLS uses a direct demeaning + Cholesky solve (no IRLS iteration) with an Irons-Tuck accelerated demeaning engine mirroring fixest's C++ implementation. Poisson is stable up to 1M+ observations.

## Installation

```julia
using Pkg
Pkg.develop(path="Panelest.jl")
```

## Quick Start

```julia
using Panelest, DataFrames

df = DataFrame(y = rand(1:10, 1000), x = randn(1000),
               id = repeat(1:100, inner=10), year = repeat(1:10, outer=100))

# OLS with two-way fixed effects
feols(df, @formula(y ~ x + fe(id) + fe(year)))

# Poisson
fepois(df, @formula(y ~ x + fe(id) + fe(year)))

# Clustered standard errors
feols(df, @formula(y ~ x + fe(id) + fe(year)), vcov = Vcov.cluster(:id))
```

## Tutorials

| Tutorial | Description |
|----------|-------------|
| [Getting Started](docs/src/tutorials/01_getting_started.md) | Installation, basic syntax, fixed effects |
| [Non-Linear Models](docs/src/tutorials/02_nonlinear_models.md) | Poisson, Logit, Probit, Conditional Logit |
| [fixest Examples](docs/src/tutorials/03_fixest_examples.md) | Side-by-side comparison with R's fixest |
| [Staggered DiD](docs/src/tutorials/04_staggered_did.md) | Difference-in-differences with staggered adoption |
| [DuckDB Integration](docs/src/tutorials/05_duckdb_integration.md) | Out-of-core estimation from database tables |
| [Benchmarking](docs/src/tutorials/06_benchmarking.md) | Performance comparison vs fixest (R/C++) |

## Instrumental Variables

```julia
# Single instrument
feiv(df, @formula(y ~ x_exo + fe(id)), endo=:x_endo, inst=:z)

# Diagnostics
model.diagnostics.first_stage_F   # first-stage F-statistic
model.diagnostics.wu_hausman      # Wu-Hausman endogeneity test
model.diagnostics.sargan          # Sargan overidentification test
```

## DuckDB (out-of-core)

```julia
using Panelest, DuckDB, DBInterface

con = DBInterface.connect(DuckDB.DB())
# DuckDB.execute(con, "CREATE TABLE data AS SELECT * FROM 'large.parquet'")

fepois(con, "data", @formula(y ~ x + fe(id) + fe(year)))
```
