using Documenter, Panelest

makedocs(
    sitename = "Panelest.jl",
    modules = [Panelest],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started"    => "tutorials/01_getting_started.md",
            "Non-Linear Models"  => "tutorials/02_nonlinear_models.md",
            "fixest Examples"    => "tutorials/03_fixest_examples.md",
            "Staggered DiD"      => "tutorials/04_staggered_did.md",
            "DuckDB Integration" => "tutorials/05_duckdb_integration.md",
            "Benchmarking"       => "tutorials/06_benchmarking.md",
            "LAUS DuckDB Demo"    => "tutorials/07_laus_duckdb_demo.md",
        ],
        "Reference" => "reference.md",
    ],
    warnonly = true,
)

deploydocs(
    repo = "github.com/xiangao/Panelest.jl.git",
    devbranch = "main",
    push_preview = false,
)
