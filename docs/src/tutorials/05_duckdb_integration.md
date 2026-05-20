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

DuckDB is useful here as a storage and preprocessing layer. You can keep a large
panel table in DuckDB, run filtering or aggregation there, and then pass the
prepared table to `Panelest.jl`.

## Why use DuckDB with Panelest?

1.  **Storage**: Keep the raw table outside Julia memory.
2.  **Compression**: Collapse repeated cells into means and counts before estimation.
3.  **Speed**: Let DuckDB do the aggregation work it is good at.

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
