# Panelest.jl Project Memory

## Naming Conventions
- Models estimated via IRLS with explicit fixed effects should use the `fe` prefix (e.g., `feols`, `fepois`, `felogit`, `feprobit`).
- Models that handle fixed effects through other statistical conditioning techniques should NOT use the `fe` prefix to distinguish their theoretical approach. Specifically:
  - **`clogit`**: Chamberlain's Conditional Logit (conditions out fixed effects using the recursive Gail, Lubin, and Gastwirth algorithm).
  - **`cre`**: Wooldridge's Correlated Random Effects (handles fixed effects by augmenting covariates with their group means).

## Current Status
- DuckDB integration successfully validated against R's `fixest` for Poisson models (`fepois`) via out-of-core data frames.
- Replaced `beta .+= delta` with `beta .= delta` in `irls.jl` to correctly calculate Newton-Raphson/IRLS steps.

## Future Implementation Notes
- When adding new statistical models, check whether they fit the `fe` prefix paradigm or the non-`fe` paradigm depending on how they handle unobserved heterogeneity.