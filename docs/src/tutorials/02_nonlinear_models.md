# Non-Linear Panel Models in Panelest.jl

`Panelest.jl` provides specialized tools for non-linear panel models, including binary outcomes and count data.

## Poisson Models

Poisson regressions with fixed effects are highly optimized using a unified IRLS-FE engine.

```julia
m = fepois(df, @formula(y_counts ~ x1 + fe(id) + fe(time)))
println(m)
```

**Output:**
```text
Panelest Model: poisson
Number of obs: 1000
Converged: true
Iterations: 15
────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
x1             8.211052    0.026452   310.42    <1e-99   8.159208   8.262896
────────────────────────────────────────────────────────────────────────
```

## Binary Choice Models

### Logit and Probit

You can estimate standard logit and probit models with high-dimensional fixed effects:

```julia
m_logit = felogit(df, @formula(y_binary ~ x1 + fe(id)))
println(m_logit)
```

**Output:**
```text
Panelest Model: logit
Number of obs: 1000
Converged: true
Iterations: 15
────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
x1            11.224818    0.091217   123.06    <1e-99  11.046037  11.403599
────────────────────────────────────────────────────────────────────────
```

Note: In models with binary outcomes and many fixed effects, be mindful of the incidental parameters problem. For logit models, Chamberlain's conditional logit (`clogit`) is often preferred for consistency.

### Chamberlain's Conditional Logit

`clogit` estimates a fixed-effects logit model using the conditional likelihood approach, which remains consistent even as the number of fixed effects grows.

```julia
# Grouped by :id
m_clogit = clogit(df, @formula(y_binary ~ x1), :id)
println(m_clogit)
```

**Output:**
```text
[ Info: Using conditional logit (Chamberlain) recursive algorithm.
Panelest Model: clogit
Number of obs: 1000
Converged: true
Iterations: 0
────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
x1             0.636319    0.085540     7.44    <1e-12   0.468664   0.803975
────────────────────────────────────────────────────────────────────────
```

`Panelest.jl` uses a recursive algorithm (Gail et al., 1981) that efficiently handles groups with any number of treated units.

## Correlated Random Effects (CRE)

For other non-linear models (like probit) where conditional logit is not available, the Correlated Random Effects approach (Wooldridge, 2010) is a robust alternative. It augments the covariates with their group-level means to capture the correlation between the fixed effects and the observed variables.

```julia
# CRE Probit
m_cre = cre(df, @formula(y_binary ~ x1), :id, family=:probit)
println(m_cre)
```

The `cre` function automatically identifies time-varying variables and computes their group means, adding them to the model before estimation.
