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
- `src/etwfe.jl` — Wooldridge (2021) Extended TWFE for staggered DiD:
  `etwfe()` fits cohort-FE + year-FE with automatic post-treatment
  cohort×year dummies, `emfx()` aggregates to overall/event/calendar
  ATTs, `dataset("mpdta")` is a bundled synthetic demo panel.
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

## Running tests

```julia
cd Panelest.jl && julia --project=. test/runtests.jl
```
No known-slow tests; full suite runs in well under a minute.

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
