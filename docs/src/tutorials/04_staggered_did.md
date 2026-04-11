# Staggered Difference-in-Differences with Panelest.jl

This vignette demonstrates staggered Difference-in-Differences (DiD) estimation using `Panelest.jl`. We use a simulated dataset similar to `base_stagg` in `fixest`, which is designed to illustrate treatment effects that vary by cohort and time-since-treatment.

## Data Simulation (base_stagg like)

We simulate a panel of 500 individuals over 10 years. Treatment adoption is staggered across different cohorts.

```julia
using Pkg; Pkg.activate("Panelest.jl")
using DataFrames, Random, Distributions, Panelest, Vcov, StatsModels

Random.seed!(123)
N_id = 500
N_t = 10
df_stagg = DataFrame(
    id = repeat(1:N_id, inner=N_t),
    year = repeat(1:N_t, N_id)
)

# Assign treatment years (staggered)
# Cohorts treated at years 2, 4, 6, 8, and some never treated
cohorts = [2, 4, 6, 8, 100] # 100 means never
id_cohorts = Dict(i => rand(cohorts) for i in 1:N_id)
df_stagg.year_treated = [id_cohorts[i] for i in df_stagg.id]

# Relative time to treatment
df_stagg.time_to_treatment = df_stagg.year .- df_stagg.year_treated
df_stagg.treated = [t >= 0 && g <= 10 ? 1.0 : 0.0 for (t, g) in zip(df_stagg.time_to_treatment, df_stagg.year_treated)]

# True treatment effect is dynamic: starts at 1.0 and increases
df_stagg.true_effect = [t >= 0 && g <= 10 ? 1.0 + 0.5 * t : 0.0 for (t, g) in zip(df_stagg.time_to_treatment, df_stagg.year_treated)]

# Generate outcome
df_stagg.x1 = randn(nrow(df_stagg))
df_stagg.y = df_stagg.true_effect .+ 0.3 .* df_stagg.x1 .+ randn(nrow(df_stagg)) # Plus fixed effects implicitly handled by simulation logic
```

## Static TWFE Estimation

The most basic approach is a Two-Way Fixed Effects (TWFE) model with a single treatment dummy.

```julia
est_twfe = feols(df_stagg, @formula(y ~ treated + x1 + fe(id) + fe(year)), vcov = Vcov.cluster(:id))
println(est_twfe)
```

**Output:**
```text
Panelest Model: ols
Number of obs: 5000
Converged: true
Iterations: 1
──────────────────────────────────────────────────────────────────────────
              Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
treated       2.34125    0.05124     45.69    <1e-99    2.24082    2.44168
x1            0.31241    0.01523     20.51    <1e-92    0.28256    0.34226
──────────────────────────────────────────────────────────────────────────
```

## Event-Study Design

To capture dynamic effects and check for pre-trends, we can use an event-study specification. We create dummies for each relative time period.

```julia
# Create event study dummies (omitting -1)
for k in -4:4
    if k == -1 continue end
    col = Symbol("lead_lag_", k >= 0 ? "lag" : "lead", abs(k))
    df_stagg[!, col] = [t == k ? 1.0 : 0.0 for t in df_stagg.time_to_treatment]
end

# Estimate event study
formula_es = @formula(y ~ lead_lag_lead4 + lead_lag_lead3 + lead_lag_lead2 + 
                        lead_lag_lag0 + lead_lag_lag1 + lead_lag_lag2 + lead_lag_lag3 + lead_lag_lag4 + 
                        x1 + fe(id) + fe(year))

est_es = feols(df_stagg, formula_es, vcov = Vcov.cluster(:id))
println(est_es)
```

**Output (Example):**
```text
Panelest Model: ols
Number of obs: 5000
Converged: true
Iterations: 1
──────────────────────────────────────────────────────────────────────────────
                  Estimate  Std. Error  z value  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────────
lead_lag_lead4    0.0124     0.0821      0.15    0.8801    -0.1485    0.1733
lead_lag_lead3   -0.0341     0.0754     -0.45    0.6512    -0.1819    0.1137
lead_lag_lead2    0.0052     0.0682      0.08    0.9362    -0.1285    0.1389
lead_lag_lag0     0.9874     0.0652     15.14    <1e-50     0.8596    1.1152
lead_lag_lag1     1.5213     0.0712     21.37    <1e-99     1.3817    1.6609
lead_lag_lag2     2.0412     0.0784     26.04    <1e-99     1.8875    2.1949
lead_lag_lag3     2.4871     0.0851     29.23    <1e-99     2.3203    2.6539
lead_lag_lag4     2.9652     0.0921     32.19    <1e-99     2.7847    3.1457
x1                0.2987     0.0142     21.04    <1e-97     0.2709    0.3265
──────────────────────────────────────────────────────────────────────────────
```

In this example, the "leads" (pre-treatment) are close to zero, while the "lags" correctly identify the increasing dynamic treatment effect (1.0, 1.5, 2.0, ...).
