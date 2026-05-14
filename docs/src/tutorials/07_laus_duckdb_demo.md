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
