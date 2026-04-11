# Benchmarking: Panelest.jl vs fixest

Performance comparison between **Panelest.jl** (Julia) and **fixest** (R/C++) using identical simulated data and model specifications from fixest's `_BENCHMARK_SIM` suite.

## Data Generation

The simulated data follows a panel structure with the following characteristics:

- **Panel structure**: 10 years, ~23 individuals per firm
- **Sample sizes**: 10K, 100K, 1M, and 10M observations
- **Replications**: 3 random seeds per sample size
- **Two DGPs**:
  - **Simple**: Random firm assignment (`firm_id = sample(1:nb_firm, n, TRUE)`)
  - **Difficult**: Deterministic firm assignment (`firm_id = rep(1:nb_firm, length.out = n)`) — creates highly correlated structure that stresses convergence

### Data Generating Process

```
y = x1 + 0.05*x1² + firm_fe + unit_fe + year_fe + ε
```

Where `x1 ~ N(0,1)`, `x2 = x1²`, all fixed effects and `ε ~ N(0,1)`. For Poisson models, `y = exp(y)`.

### Fixed Effects Specifications

```
# 2 FEs:
y ~ x1 + x2 | indiv_id + firm_id              (simple)
y ~ x1 + x2 | indiv_id + firm_id_difficult    (difficult)

# 3 FEs:
y ~ x1 + x2 | indiv_id + firm_id + year              (simple)
y ~ x1 + x2 | indiv_id + firm_id_difficult + year    (difficult)
```

## Results: OLS

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

## Why Panelest Performs Well

1. **Ported fixest algorithms**: The core C++ logic from fixest — Irons-Tuck acceleration and concentrated likelihood — is directly ported to Julia, eliminating the algorithmic gap.

2. **Optimized demeaning engine**: `DemeanSolver` uses pre-allocated buffers and a specialized 2-FE cell structure (`TwoFEGauSolver`) that reduces each Gauss-Seidel sweep from O(n) to O(cells), where cells is the number of unique (i, j) pairs.

3. **Julia's LLVM compilation**: Julia generates highly optimized machine code for specific types (`Int32` refs, `Float64` weights), allowing the pure Julia implementation to match or exceed C++ performance.

4. **Concentrated likelihood for Poisson**: Solving for optimal fixed effects in an inner loop using a closed-form multiplicative update is significantly more efficient than standard IRLS.

## Implementation Comparison

| | fixest | Panelest.jl |
|-|--------|-------------|
| Language | R + C++ (Rcpp) | Pure Julia |
| FE absorption | Custom C++ (Irons-Tuck) | Ported Julia (Irons-Tuck) |
| Poisson solver | Berge concentrated likelihood | Concentrated likelihood + 2-FE cell trick |
| Parallelism | Multi-threaded C++ | Julia `Threads.@threads` |

## Trade-offs

| | fixest | Panelest.jl |
|-|--------|-------------|
| Maturity | Highly mature, production-ready | Newer, active development |
| Features | Rich (stepwise, DiD, export) | Core estimators |
| Ecosystem | R (visualization, tables) | Julia (composability, speed) |

## Methodology

- **Timing**: R uses `Sys.time()`; Julia uses `time()` with `GC.gc()` called before each run
- **Convergence**: Both packages use tolerance ~1e-8 with up to 10,000 FE iterations
- **Replications**: 3 seeds per configuration; tables show averages

## Reproducing

```bash
cd Panelest.jl/benchmark/
julia setup_data.jl   # generate data matching fixest's _BENCHMARK_SIM
bash run_all.sh       # run Julia + R benchmarks
```

Results land in `benchmark/results/`.

## References

- Bergé, L. (2018). Efficient estimation of maximum likelihood models with multiple fixed-effects. *CRAN*.
- Benchmark design follows fixest's `_BENCHMARK_SIM` suite.
