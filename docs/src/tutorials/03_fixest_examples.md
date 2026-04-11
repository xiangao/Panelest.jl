# Replication of fixest Examples in Panelest.jl

This vignette replicates some of the core examples from the R package `fixest` walkthrough using `Panelest.jl`. We use simulated data that mimics the structure of the `trade` and `base_did` datasets provided in `fixest`.

## Trade Data Example (Gravity Model)

The `trade` dataset is a sample of bilateral trade between EU countries. We estimate a simple gravity model to find the effect of geographic distance on trade volume, controlling for origin, destination, product, and year fixed effects.

### Data Simulation

```julia
using Pkg; Pkg.activate("Panelest.jl")
using DataFrames, Random, Distributions, Panelest, StatsModels

Random.seed!(42)
N = 5000
origins = ["FR", "DE", "IT", "ES", "NL", "BE", "AT", "IE", "PT", "FI", "SE", "DK", "GR", "LU", "GB"]
destinations = origins
products = 1:20
years = 2007:2016

df = DataFrame(
    Origin = rand(origins, N),
    Destination = rand(destinations, N),
    Product = rand(products, N),
    Year = rand(years, N),
    dist_km = exp.(rand(Normal(7, 1), N))
)

df.log_dist = log.(df.dist_km)

# Latent eta with distance effect of -0.8
eta = -0.8 .* df.log_dist .+ randn(N)
df.Euros = [rand(Poisson(exp(e))) for e in eta]
```

### Poisson Estimation

We use `fepois` to estimate the model. Fixed effects are automatically handled, and the intercept is suppressed when `fe()` terms are present, matching `fixest` behavior.

```julia
gravity_pois = fepois(df, @formula(Euros ~ log_dist + fe(Origin) + fe(Destination) + fe(Product) + fe(Year)))
println(gravity_pois)
```

**Output:**
```text
Panelest Model: poisson
Number of obs: 5000
Converged: true
Iterations: 18
─────────────────────────────────────────────────────────────────────────
             Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
log_dist    -13.5298    0.0802     -168.65    <1e-99   -13.6870   -13.3725
─────────────────────────────────────────────────────────────────────────
```

### OLS Estimation (Log-Log)

Similarly, we can estimate the same relationship using OLS by taking the logarithm of the dependent variable.

```julia
df.log_Euros = log.(df.Euros .+ 1)
gravity_ols = feols(df, @formula(log_Euros ~ log_dist + fe(Origin) + fe(Destination) + fe(Product) + fe(Year)))
println(gravity_ols)
```

**Output:**
```text
Panelest Model: ols
Number of obs: 5000
Converged: true
Iterations: 1
─────────────────────────────────────────────────────────────────────────
             Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
log_dist    -0.0335     0.0143      -2.35    0.0187    -0.0614    -0.0056
─────────────────────────────────────────────────────────────────────────
```

## Difference-in-Differences Example

The `base_did` dataset in `fixest` is used to illustrate basic DiD estimation.

### Data Simulation

```julia
# 104 individuals, 10 periods
N_id = 104
N_t = 10
df_did = DataFrame(
    id = repeat(1:N_id, inner=N_t),
    period = repeat(1:N_t, N_id)
)

df_did.treat = [i > N_id/2 ? 1.0 : 0.0 for i in df_did.id]
df_did.post = [t > 5 ? 1.0 : 0.0 for t in df_did.period]
df_did.x1 = randn(nrow(df_did))

# Outcome with a true ATT of 2.0
df_did.y = 2.0 .* df_did.treat .* df_did.post .+ 0.5 .* df_did.x1 .+ randn(nrow(df_did))
```

### DiD Estimation with Clustering

We estimate the DiD effect using `feols` and cluster standard errors at the individual level using `Vcov.cluster`.

```julia
using Vcov
est_did = feols(df_did, @formula(y ~ x1 + treat&post + fe(id) + fe(period)), vcov = Vcov.cluster(:id))
println(est_did)
```

**Output (Example):**
```text
Panelest Model: ols
Number of obs: 1040
Converged: true
Iterations: 1
──────────────────────────────────────────────────────────────────────────
              Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
x1            0.48512    0.03125     15.52    <1e-53    0.42387    0.54637
treat & post  1.98741    0.06214     31.98    <1e-99    1.86562    2.10920
──────────────────────────────────────────────────────────────────────────
```
