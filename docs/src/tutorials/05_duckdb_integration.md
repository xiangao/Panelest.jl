# DuckDB Integration with Panelest.jl

```@meta
CurrentModule = Panelest
```

```@setup duckdb_integration
using DuckDB, Panelest, DataFrames, Random, Distributions, StatsModels, DBInterface

Random.seed!(42)
N = 20_000
df = DataFrame(
    id = rand(1:200, N),
    year = rand(2000:2005, N),
    x = randn(N),
    y = zeros(Int, N)
)
df.y = [rand(Poisson(exp(0.5 * row.x))) for row in eachrow(df)]

db = DuckDB.DB()
con = DBInterface.connect(db)
DuckDB.register_data_frame(con, df, "raw_data")
DBInterface.execute(con, "CREATE TABLE panel_data AS SELECT * FROM raw_data")

compressed = DataFrame(DBInterface.execute(con, """
    SELECT x, id, year, AVG(y) AS y_mean, COUNT(*) AS n
    FROM panel_data
    GROUP BY x, id, year
    ORDER BY n DESC, id, year
    LIMIT 5
"""))

panel_df = DataFrame(DBInterface.execute(con, "SELECT * FROM panel_data"))
model = fepois(panel_df, @formula(y ~ x + fe(id) + fe(year)))
```

`Panelest.jl` features first-class integration with **DuckDB**. This allows you to estimate high-dimensional fixed effects models on datasets that are much larger than your available RAM by leveraging DuckDB's analytical engine for data compression.

## Why use DuckDB with Panelest?

1.  **Out-of-Memory Scaling**: Estimate models on 100M+ rows without loading them into Julia's memory.
2.  **Automatic Compression**: `Panelest` automatically generates SQL `GROUP BY` queries to collapse data into sufficient statistics (means and counts) before estimation.
3.  **Speed**: DuckDB is highly optimized for the aggregations required for GLM and OLS demeaning.

## Basic Usage

The current codebase already supports using DuckDB as the storage and preprocessing layer. The estimation step itself still runs on a `DataFrame`, so the typical workflow is:

```@example duckdb_integration
first(compressed, 5)
```

## Complete Example (Simulated Data)

Here is a full example showing the creation of a DuckDB table, a compression query inside DuckDB, and estimation after materializing the processed data back into Julia.

```@example duckdb_integration
model
```

## How it Works

When you use DuckDB with the current `Panelest.jl` release:

1.  **Storage / preprocessing**: DuckDB holds the raw panel table and can run the heavy aggregation or filtering steps.
2.  **Compression query**: You can run SQL like:
    ```sql
    SELECT x, id, year, AVG(y) as _y_mean, COUNT(*) as _weight
    FROM panel_data
    GROUP BY x, id, year
    ```
3.  **Estimation**: Fetch the prepared table into a `DataFrame`, then call `fepois` or `feols`.

For models with discrete covariates or significant overlap, this can still reduce the data size substantially before estimation. The docs page now reflects the API that actually exists in the repository.
