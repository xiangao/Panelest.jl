# Benchmarking: Panelest.jl vs fixest

Performance comparison between **Panelest.jl** (Julia) and **fixest** (R/C++) using identical simulated data and model specifications from fixest's `_BENCHMARK_SIM` suite.

## Data Generating Process

Panel structure with 10 years and ~23 individuals per firm:

```
y = x1 + 0.05*x1² + firm_fe + unit_fe + year_fe + ε
```

where `x1 ~ N(0,1)`, `x2 = x1²`, all effects and `ε ~ N(0,1)`.

Two DGPs stress-test convergence differently:
- **Simple**: random firm assignment
- **Difficult**: deterministic (`firm_id = rep(1:nb_firm, length.out = n)`) — creates highly correlated FE structure

For Poisson, the outcome is exponentiated: `y = exp(y)`.

## Results: OLS

Models: `y ~ x1 + x2 | indiv_id + firm_id` (2 FE) and `y ~ x1 + x2 | indiv_id + firm_id + year` (3 FE).

**Difficult DGP, 2 fixed effects:**

| Observations | fixest | Panelest.jl | Ratio |
|--------------|--------|-------------|-------|
| 10,000       | 0.01s  | 0.003s      | **0.3x** ✅ |
| 100,000      | 0.10s  | 0.03s       | **0.3x** ✅ |
| 1,000,000    | 2.0s   | 0.37s       | **0.2x** ✅ |
| 10,000,000   | 21.6s  | 18.5s       | **0.9x** ✅ |

**Simple DGP, 3 fixed effects:**

| Observations | fixest | Panelest.jl | Ratio |
|--------------|--------|-------------|-------|
| 10,000       | 0.01s  | 0.004s      | **0.4x** ✅ |
| 100,000      | 0.02s  | 0.03s       | 1.5x |
| 1,000,000    | 0.23s  | 0.57s       | 2.5x |
| 10,000,000   | 2.3s   | 2.1s        | **0.9x** ✅ |

## Results: Poisson

**Simple DGP, 2 fixed effects:**

| Observations | fixest | Panelest.jl | Ratio |
|--------------|--------|-------------|-------|
| 10,000       | 0.04s  | 0.03s       | **0.8x** ✅ |
| 100,000      | 0.34s  | 0.51s       | 1.5x |
| 1,000,000    | 4.3s   | 6.4s        | 1.5x |

## Results: Logit

**Simple DGP, 2 fixed effects:**

| Observations | fixest | Panelest.jl | Ratio |
|--------------|--------|-------------|-------|
| 10,000       | 0.04s  | 0.03s       | **0.8x** ✅ |
| 100,000      | 1.1s   | 0.33s       | **0.3x** ✅ |

## Implementation

| | fixest | Panelest.jl |
|-|--------|-------------|
| Language | R + C++ (Rcpp) | Pure Julia |
| FE absorption | Custom C++ (Irons-Tuck) | Ported Julia (Irons-Tuck) |
| Poisson solver | Berge concentrated likelihood | Concentrated likelihood + 2-FE cell trick |
| Parallelism | Multi-threaded C++ | Julia `Threads.@threads` |

## Reproducing

```bash
cd Panelest.jl/benchmark/
julia setup_data.jl   # generate data
bash run_all.sh       # run Julia + R benchmarks
```

Results land in `benchmark/results/`.
