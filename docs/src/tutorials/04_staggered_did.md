# Staggered Difference-in-Differences with Panelest.jl

```@meta
CurrentModule = Panelest
```

This vignette demonstrates staggered Difference-in-Differences (DiD) estimation using `Panelest.jl`. We use a simulated dataset similar to `base_stagg` in `fixest`, which is designed to illustrate treatment effects that vary by cohort and time-since-treatment.

## Data Simulation (base_stagg like)

We simulate a panel of 500 individuals over 10 years. Treatment adoption is staggered across different cohorts.

```@example staggered_did
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

```@example staggered_did
est_twfe = feols(df_stagg, @formula(y ~ treated + x1 + fe(id) + fe(year)), vcov = Vcov.cluster(:id))
println(est_twfe)
```

## Event-Study Design

To capture dynamic effects and check for pre-trends, we can use an event-study specification. We create dummies for each relative time period.

```@example staggered_did
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

In this example, the "leads" (pre-treatment) are close to zero, while the "lags" correctly identify the increasing dynamic treatment effect (1.0, 1.5, 2.0, ...).
