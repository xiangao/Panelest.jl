# Panelest.jl

A high-performance Julia package for estimating panel data models with high-dimensional fixed effects. It is designed to be fast and memory-efficient, supporting out-of-core processing through databases like DuckDB.

Currently supports:
- OLS (`feols`)
- Poisson (`fepois`)
- Logit (`felogit`)
- Probit (`feprobit`)
- Conditional Logit (`clogit`)
- Correlated Random Effects (`cre`)

## Installation

```julia
using Pkg
Pkg.develop(path="Panelest.jl") # If working locally
```

## Usage

Panelest integrates seamlessly with `DataFrames.jl` and database backends like `DuckDB.jl`.

### Basic Example with DataFrames

```julia
using Panelest, DataFrames

# Create dummy data
df = DataFrame(y = rand(1:10, 100), x = randn(100), id = repeat(1:10, inner=10), year = repeat(1:10, outer=10))

# Estimate a Poisson model with fixed effects
model = fepois(df, @formula(y ~ x + fe(id) + fe(year)))
println(model)
```

### Out-of-Core Processing with DuckDB

Panelest supports estimating models directly from DuckDB tables, allowing you to fit models on datasets much larger than RAM without loading them entirely into Julia's memory:

```julia
using Panelest, DuckDB, DBInterface

db = DuckDB.DB()
con = DBInterface.connect(db)

# Load or register your large dataset in DuckDB
# DuckDB.execute(con, "CREATE TABLE panel_data AS SELECT * FROM 'large_data.csv'")

model_duck = fepois(con, "panel_data", @formula(y ~ x + fe(id) + fe(year)))
println(model_duck)
```
