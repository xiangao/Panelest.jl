# LAUS DuckDB Demo

This repository includes a local demo using the Bureau of Labor Statistics Local Area
Unemployment Statistics files. It is designed to show `Panelest.jl` on a real
county-month panel while DuckDB handles the raw tab-delimited files.

The demo expects the LAUS directory at `~/projects/data/laus`. You can override that
with `LAUS_DIR`.

```bash
julia --project=docs examples/laus_duckdb_demo.jl
```

```bash
LAUS_DIR=/path/to/laus julia --project=docs examples/laus_duckdb_demo.jl
```

## What the Demo Does

The script reads:

- `la.series` for the mapping from `series_id` to area, measure, and seasonality.
- `la.area` for county names and area types.
- `la.data.64.County` for county-level monthly LAUS observations.

DuckDB filters the raw county file, joins the metadata, and pivots the four county
measures into a panel:

- unemployment rate
- unemployment
- employment
- labor force

The resulting table is called `laus_panel`. For the default 2015-2025 window it has
421,813 county-month rows and 3,222 counties in the current local LAUS snapshot.

## How DuckDB Is Used

The LAUS source files remain plain tab-delimited text files. The demo does not first
load the raw county file into a Julia `DataFrame`. Instead, DuckDB scans the raw
files directly:

```julia
FROM read_csv_auto(".../la.data.64.County", delim='\t', header=true, all_varchar=true)
```

DuckDB then does the relational work:

1. Reads `la.series` and `la.area` as metadata views.
2. Reads `la.data.64.County` as a raw county observation view.
3. Filters to monthly rows, the requested year range, county areas, unadjusted
   series, and measures `03`-`06`.
4. Joins the raw observations to the metadata.
5. Pivots the long BLS measure rows into one county-month row with unemployment
   rate, unemployment, employment, and labor force columns.
6. Stores the prepared analysis table as `laus_panel` inside DuckDB.

Only after that does `Panelest.jl` receive data for estimation. The call

```julia
feols(con, "laus_panel", @formula(unemp_rate ~ log_labor_force + fe(county_id) + fe(year_month)))
```

passes a DuckDB connection and table name to the extension. The extension asks
DuckDB for the formula columns, creates `_y_mean` and `_weight` sufficient-statistic
columns with a SQL `GROUP BY`, materializes the compressed table into Julia, and
then runs the standard `feols` estimator.

For this particular LAUS model, `log_labor_force` is nearly continuous, so the
compression step does not shrink the data much. DuckDB's main contribution here is
fast raw-file scanning, filtering, joining, and reshaping before Julia sees the
analysis table.

## Model

The demo estimates two descriptive fixed-effects specifications through the DuckDB
extension:

```julia
model_rate = feols(
    con,
    "laus_panel",
    @formula(unemp_rate ~ log_labor_force + fe(county_id) + fe(year_month)),
)

model_count = feols(
    con,
    "laus_panel",
    @formula(log_unemployment ~ log_labor_force + fe(county_id) + fe(year_month)),
)
```

These are not causal designs. They are compact demonstrations of:

- reading large public data directly through DuckDB,
- using SQL to build an analysis table,
- calling `Panelest.feols` on a DuckDB table,
- absorbing county and year-month fixed effects.

You can change the time window without editing the file:

```bash
LAUS_START_YEAR=1990 LAUS_END_YEAR=2025 julia --project=docs examples/laus_duckdb_demo.jl
```

## R Baseline Without DuckDB

The repository also includes an R baseline that does not use DuckDB:

```bash
Rscript examples/laus_r_fixest_demo.R
```

It uses `data.table::fread()` to read the same raw LAUS flat files, reshapes the
long BLS measure rows into the same county-month panel, and estimates the same
fixed-effect regressions with `fixest::feols()`:

```r
feols(unemp_rate ~ log_labor_force | county_id + year_month, data = laus_panel)
feols(log_unemployment ~ log_labor_force | county_id + year_month, data = laus_panel)
```

This is a useful comparison because it separates two pieces of the workflow:

- **Panel construction**: DuckDB vs. R/data.table reading and reshaping flat files.
- **Estimation**: `Panelest.feols` vs. `fixest::feols` on the same regression.

## Example Output

On the local LAUS snapshot at `~/projects/data/laus`, the default 2015-2025
run produces:

```text
LAUS county-month panel
1x5 DataFrame
 Row | rows    counties  first_year  last_year  avg_unemp_rate
     | Int64   Int64     Int32       Int32      Float64
-----+---------------------------------------------------------
   1 | 421813      3222        2015       2025            4.73

Model 1: county unemployment rate on labor-force size, county FE, and year-month FE
Panelest Model: ols
Number of obs: 421813
Converged: true
Iterations: 1
Estimate for log_labor_force: -3.1717
Std. Error: 0.0256071

Model 2: log unemployed workers on labor-force size, county FE, and year-month FE
Panelest Model: ols
Number of obs: 421813
Converged: true
Iterations: 1
Estimate for log_labor_force: 0.321523
Std. Error: 0.0256071

Coefficients
2x3 DataFrame
 Row | model              coefficient  std_error
     | String             Float64      Float64
-----+-------------------------------------------
   1 | unemployment rate    -3.1717    0.0256071
   2 | log unemployment      0.321523  0.0256071

Timing
4x2 DataFrame
 Row | step                     seconds
     | String                   Float64
-----+----------------------------------
   1 | DuckDB panel build          3.99
   2 | feols unemployment rate     6.0
   3 | feols log unemployment      0.5
   4 | total in-process           14.62
```

The same run took 21.56 seconds by shell wall clock. The difference is Julia
startup, package loading, and related overhead outside the timed body of the script.
The first `feols` call also includes method compilation, which is why the second
model is much faster in the same process. Timings vary by machine, storage, and
whether Julia has already precompiled the relevant packages.

The R/data.table/fixest baseline produced the same panel size and same point
estimates:

```text
LAUS county-month panel
     rows counties first_year last_year avg_unemp_rate
1: 421813     3222       2015      2025           4.73

Coefficients
               model coefficient   std_error
1: unemployment rate  -3.1717005 0.029632306
2:  log unemployment   0.3215226 0.004648597

Timing
                       step seconds
1:   data.table panel build   10.00
2: fixest unemployment rate    0.07
3:  fixest log unemployment    0.06
4:         total in-process   10.19
shell_wall_seconds 11.61
```

In this local run, DuckDB built the analysis table faster than the R flat-file
pipeline (`3.99` seconds vs. `10.00` seconds). The R script was faster end to end
because `fixest` estimated these two regressions much faster and R had less
startup/package-loading overhead in this run. The point estimates agree; the
reported standard errors differ because the two demos currently use their
packages' default IID variance calculations after different internal estimation
paths.

The documentation shows captured output instead of executing the LAUS script during
the docs build, because GitHub Actions does not have the local BLS files.
