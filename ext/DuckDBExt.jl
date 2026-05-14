module DuckDBExt

using Panelest
using DataFrames
using StatsModels
using DuckDB
using DBInterface

"""
    _duck_compress(con, table, formula)

Helper to generate and execute DuckDB SQL for data compression.
Groups by all RHS variables (including fixed effects).
"""
function _duck_compress(con, table, formula)
    formula_main, formula_fe = Panelest.parse_fe(formula)
    
    # Identify all columns needed
    # LHS
    lhs_var = string(formula.lhs)
    
    # RHS (covariates + FEs)
    rhs_vars = String[]
    # Covariates
    append!(rhs_vars, string.(StatsModels.termvars(formula_main.rhs)))
    # Fixed Effects
    for term in Panelest.eachterm(formula_fe.rhs)
        if term isa Panelest.FixedEffectTerm
            push!(rhs_vars, string(term.x))
        elseif term isa FunctionTerm && term.f === Panelest.fe
            push!(rhs_vars, string(term.args[1]))
        end
    end
    
    unique_vars = unique(rhs_vars)
    group_cols = join(unique_vars, ", ")
    
    # Build query
    # We use SUM(y) as total outcome and COUNT(*) as weight
    query = """
    SELECT $group_cols, 
           CAST(SUM($lhs_var) AS DOUBLE) / COUNT(*) as _y_mean, 
           CAST(COUNT(*) AS DOUBLE) as _weight
    FROM $table
    GROUP BY $group_cols
    """
    
    df_compressed = DataFrame(DBInterface.execute(con, query))
    
    # Create a new formula pointing to the mean outcome
    new_formula = FormulaTerm(Term(Symbol("_y_mean")), formula.rhs)
    
    return df_compressed, new_formula
end

function Panelest.feols(con::DuckDB.Connection, table::AbstractString, formula::FormulaTerm; kwargs...)
    df, f = _duck_compress(con, table, formula)
    return feols(df, f; weights=:_weight, kwargs...)
end

function Panelest.fepois(con::DuckDB.Connection, table::AbstractString, formula::FormulaTerm; kwargs...)
    df, f = _duck_compress(con, table, formula)
    return fepois(df, f; weights=:_weight, kwargs...)
end

function Panelest.felogit(con::DuckDB.Connection, table::AbstractString, formula::FormulaTerm; kwargs...)
    df, f = _duck_compress(con, table, formula)
    return felogit(df, f; weights=:_weight, kwargs...)
end

function Panelest.feprobit(con::DuckDB.Connection, table::AbstractString, formula::FormulaTerm; kwargs...)
    df, f = _duck_compress(con, table, formula)
    return feprobit(df, f; weights=:_weight, kwargs...)
end

end # module
