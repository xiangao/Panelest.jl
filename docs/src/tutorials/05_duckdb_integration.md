# DuckDB Integration with Panelest.jl

```@meta
CurrentModule = Panelest
```

`Panelest.jl` features first-class integration with **DuckDB**. This allows you to estimate high-dimensional fixed effects models on datasets that are much larger than your available RAM by leveraging DuckDB's analytical engine for data compression.

## Why use DuckDB with Panelest?

1.  **Out-of-Memory Scaling**: Estimate models on 100M+ rows without loading them into Julia's memory.
2.  **Automatic Compression**: `Panelest` automatically generates SQL `GROUP BY` queries to collapse data into sufficient statistics (means and counts) before estimation.
3.  **Speed**: DuckDB is highly optimized for the aggregations required for GLM and OLS demeaning.

## Basic Usage

To use this feature, simply pass a `DuckDB.Connection` and the name of your table instead of a `DataFrame`.

```julia
using DuckDB, Panelest, StatsModels
using DBInterface

# 1. Connect to DuckDB
db = DuckDB.DB("my_data.duckdb")
con = DBInterface.connect(db)

# 2. Run fepois directly on the database table
# Panelest will handle the SQL compression internally
model = fepois(con, "large_panel_table", @formula(y ~ x1 + x2 + fe(id) + fe(year)))

println(model)
```

## Complete Example (Simulated Data)

Here is a full example showing the creation of a DuckDB table and the automatic compression workflow.

```julia
using DuckDB, Panelest, DataFrames, Random, Distributions, StatsModels, DBInterface

# Generate 100,000 rows of data
Random.seed!(42)
N = 100_000
df = DataFrame(
    id = rand(1:500, N),
    year = rand(2000:2020, N),
    x = randn(N),
    y = zeros(N)
)
# True Poisson model
df.y = [rand(Poisson(exp(0.5 * row.x))) for row in eachrow(df)]

# Create a DuckDB table
db = DuckDB.DB() # In-memory database
con = DBInterface.connect(db)
DuckDB.register_data_frame(con, df, "raw_data")
DBInterface.execute(con, "CREATE TABLE panel_data AS SELECT * FROM raw_data")

# --- AUTO-COMPRESSION WORKFLOW ---

println("Running Poisson regression on 100,000 rows via DuckDB...")
model = fepois(con, "panel_data", @formula(y ~ x + fe(id) + fe(year)))

println(model)
```

## How it Works

When you pass a DuckDB connection to a `Panelest` function:

1.  **Analysis**: `Panelest` inspects the formula to find all required covariates and fixed effect dimensions.
2.  **SQL Generation**: It generates a SQL query similar to:
    ```sql
    SELECT x, id, year, AVG(y) as _y_mean, COUNT(*) as _weight
    FROM panel_data
    GROUP BY x, id, year
    ```
3.  **Weighted Estimation**: It fetches the resulting summary table and runs the core `irls_fit` engine using `_weight` as frequency weights.

For models with discrete covariates or significant overlap, this can reduce the data size by 90-99% while producing mathematically identical results to running on the full dataset.
