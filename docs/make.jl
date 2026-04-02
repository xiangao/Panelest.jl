using Documenter, Panelest

makedocs(
    sitename = "Panelest.jl",
    modules = [Panelest],
    pages = [
        "Home" => "index.md",
        "Vignettes" => [
            "Getting Started" => "vignettes/01_getting_started.md",
            "Non-Linear Models" => "vignettes/02_nonlinear_models.md",
            "fixest Examples" => "vignettes/03_fixest_examples.md",
            "Staggered DiD" => "vignettes/04_staggered_did.md",
            "DuckDB Integration" => "vignettes/05_duckdb_integration.md"
        ],
        "Reference" => "reference.md"
    ],
    remotes = nothing,
    repo = "",
    warnonly = true
)
