# Panelest.jl — project notes for Claude

Julia analogue of R's `fixest` workflow: OLS, IV/2SLS, Poisson, logit,
probit, conditional logit, correlated random effects, and Wooldridge
ETWFE — all with high-dimensional fixed-effect absorption. Can also read
from DuckDB for out-of-core estimation.

## What's where

- `src/ols.jl`, `src/poisson.jl`, `src/logit.jl`, `src/probit.jl` — the
  `feols`/`fepois`/`felogit`/`feprobit`/`feglm` estimators.
- `src/iv.jl` — `feiv` (2SLS), with first-stage F, Wu-Hausman, and Sargan
  diagnostics on the returned model.
- `src/clogit.jl` — `clogit` (Chamberlain conditional logit, recursive
  algorithm for panel choice data).
- `src/cre.jl` — `cre` (correlated random effects / Mundlak device).
- `src/direct_fe.jl`, `src/fe_convergence.jl`, `src/irls.jl` — the shared
  fixed-effects demeaning (Irons-Tuck accelerated) and IRLS machinery
  used by all of the above.
- `src/etwfe.jl` — Wooldridge (2021, 2023) Extended TWFE for staggered DiD,
  same specification as R `etwfe` 0.6: cohort + year effects, one dummy per
  treatment cell, controls with Wooldridge's interactions; `emfx()` averages
  per-observation effects into simple/group/event/calendar ATTs (count scale
  for Poisson). `dataset("mpdta")` is a bundled synthetic demo panel.
- `test/runtests.jl` — one big `@testset` per model family plus an
  `"ETWFE"` block.

All results are `PanelestModel <: StatisticalModel` with the standard
StatsAPI (`coef`, `vcov`, `stderror`, `coeftable`, ...).

## `etwfe()` — read this before touching it

Every treated cohort in the data MUST have at least one pre-treatment
period (`gvar > minimum(tvar)`). A cohort treated at or before the first
sample period isn't identified under the cohort-FE+year-FE
specification — its cohort fixed effect can't be separated from its own
post-treatment dummies, and it can't serve as a control either (it's
treated in every observed period). `etwfe()` now detects this and drops
such cohorts with a `@warn`, mirroring how Callaway & Sant'Anna (2021)
drop always-treated units.

This was a real, silent bug until 2026-07-01: `dataset("mpdta")`'s
earliest cohort (2004) coincided with the first sample year, so its
cells were silently ≈0 instead of the true 0.30 effect, and any
`emfx()` aggregate that pooled across cohorts (the overall ATT, the
early event-time buckets) came out biased. The demo dataset was fixed
by extending it to start a year earlier (2003–2007, matching the real
`mpdta`'s range) so every cohort has a genuine pre-period. If you ever
see `etwfe()` results that look suspiciously close to zero for a subset
of cohorts, check for this first.

`emfx(model::PanelestModel; gvar, tvar, ...)` (the "legacy" string-
pattern overload, matching raw `feols`/`fepois` coefficient names like
`"first_treat_str: 2006 & year_str: 2007"`) still exists for
manually-constructed ETWFE formulas, but `etwfe()` + `emfx(::ETWFEResult)`
is preferred for new code — it validates cohort identification and is
the one covered by regression tests.

Beware: a separate now-archived package (`xiangao/DiD.jl` /
`ETWFE.jl`) duplicated this dataset+API surface (including a fake
`att_gt` stub that never estimated anything) and exported a conflicting
`emfx`/`dataset`. It's deprecated in favor of this package — don't
resurrect logic from it without re-verifying against the bug above.

## `etwfe()` matches R `etwfe` (2026-10-03, branch `etwfe-covariates`)

Before this, `etwfe()` entered controls linearly only, `emfx()` weighted cells
equally, and Poisson effects were log-scale only. On Wooldridge's simulated
panel the ATT was 3.448 against R's 3.672 (truth 3.677). Now:

- Controls: x, x × cohort dummies, x × year dummies (ref = gref, tref), and
  each treatment cell × (x demeaned within cohort, over the full data).
- `cgroup = "notyet"` (default) or `"never"` (cells for every t ≠ g - 1, so
  pre-trends are free). `gref`/`tref` defaults follow R.
- Gaussian fits `feols` with cohort and year FE. Poisson fits `fepois` with an
  intercept and explicit cohort/year dummies (R's `fe = "none"`), because the
  count-scale delta method needs the full design row.
- `emfx()` follows R's row selection and averages exp(η) - exp(η - δ) (Poisson,
  `scale = "response"`, default) or δ (gaussian, or `scale = "link"`), weighted
  by observation. Pre-period event times with no free cell report SE = NaN (R: NA).
- `test/test_etwfe_r.jl` checks 19 aggregate tables against R
  (`test/data/make_etwfe_ref.R` writes fixtures + reference values): estimates
  agree to 1e-12 (Poisson with a control 1e-8), SEs to 1e-6 relative (5e-5 in that
  Poisson case, R's numerical Jacobian).

## Running tests

```julia
cd Panelest.jl && julia --project=. test/runtests.jl
```
No known-slow tests; full suite runs in about 1.5 minutes with 6 threads.

## Docs

Documenter.jl, `docs/make.jl`, deployed via `.github/workflows/docs.yml`
to <https://xiangao.github.io/Panelest.jl/dev/>. Tutorials under
`docs/src/tutorials/` include a staggered-DiD vignette — note that
vignette uses plain event-study TWFE (leads/lags dummies), NOT
`etwfe()`, so it doesn't demonstrate the cohort-identification issue
above; don't assume it validates `etwfe()`.

No `Manifest.toml` is committed (correct for a Julia library — a
committed Manifest resolved on one machine's Julia can silently break
CI on other Julia versions in the test matrix, as happened to several
sibling packages in this portfolio in 2026-07).

## Standard-error fixes (2026-10-01, branch `fix-vcov-dof`)

Three bugs, found while porting the DiD-with-continuous-treatment book chapter:

1. **Race in one-FE (and 3+ FE) demeaning.** `solve_residuals_fixest!` threads over columns
   with one shared `DemeanSolver`; `get_level_weights!` zero-filled and re-accumulated the
   solver's `cached_level_weights` from every thread at once (its `cached_weights === weights`
   check could never pass). With 6 threads, repeated identical `feols(y ~ x + fe(t))` calls
   returned x coefficients from -6.2 to 8.3 (truth 0.48). Two-FE models use `TwoFEGauSolver`
   and were never affected. Fix: level weights are computed once before the threaded loop and
   passed read-only (`get_level_weights`, no `!`). The unused cache fields remain in the struct.
2. **Simple vcov had no sigma^2.** `vcov_panelest` returned `pinv(X'X)` for `Vcov.simple()`,
   the default of `feols`/`feiv`/`etwfe`. Now scaled by RSS/(n - K) for OLS and IV
   (`scale_simple = true`); Poisson/logit/probit keep the inverse information.
3. **Small-sample K ignored the fixed effects.** Robust and clustered vcov used dof = n - p.
   Now K = p + `fe_dof(fes, clusters)`, fixest's `ssc(fixef.K = "nested")` rule:
   1 + sum over FE not nested in a cluster of (levels - 1). This equals Stata
   `xtreg, fe vce(cluster)`. `feols`'s `df_residual` uses the same count (it subtracted
   `length(fes)` before).

Tests: "OLS standard errors match fixest" pins iid, HC1 and clustered SEs (no FE, one FE,
two FE) and one-FE coefficients against R fixest values on a deterministic panel, plus a
30-call determinism check. Run with `JULIA_NUM_THREADS=6` (the race needs threads).
