# Panelest.jl Project Memory

## Naming Conventions
- Models estimated via IRLS with explicit fixed effects should use the `fe` prefix (e.g., `feols`, `fepois`, `felogit`, `feprobit`).
- Models that handle fixed effects through other statistical conditioning techniques should NOT use the `fe` prefix to distinguish their theoretical approach. Specifically:
  - **`clogit`**: Chamberlain's Conditional Logit (conditions out fixed effects using the recursive Gail, Lubin, and Gastwirth algorithm).
  - **`cre`**: Wooldridge's Correlated Random Effects (handles fixed effects by augmenting covariates with their group means).

## Current Status
- DuckDB integration successfully validated against R's `fixest` for Poisson models (`fepois`) via out-of-core data frames.
- Replaced `beta .+= delta` with `beta .= delta` in `irls.jl` to correctly calculate Newton-Raphson/IRLS steps.
- IV/2SLS estimation (`feiv`) added in `src/iv.jl`. Supports multiple endogenous variables, arbitrary instruments, FE absorption, and robust/clustered SEs. Diagnostics: first-stage F-stat, Wu-Hausman endogeneity test, Sargan overidentification test. 29 tests pass.

## Architecture
- `src/Panelest.jl` — Module, formula parsing, PanelestModel struct, StatsAPI methods
- `src/irls.jl` — Core IRLS engine, FE absorption via FixedEffects.jl, Vcov integration
- `src/ols.jl` — feols (OLS with FE)
- `src/iv.jl` — feiv (IV/2SLS): IVModel struct, 2SLS algorithm, diagnostics
- `src/poisson.jl`, `logit.jl`, `probit.jl` — GLM estimators
- `src/clogit.jl` — Chamberlain's Conditional Logit
- `src/cre.jl` — Correlated Random Effects

## Future Implementation Notes
- When adding new statistical models, check whether they fit the `fe` prefix paradigm or the non-`fe` paradigm depending on how they handle unobserved heterogeneity.