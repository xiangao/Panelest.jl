# Wooldridge's Correlated Random Effects (CRE)
# This model augments covariates with their group means to handle fixed effects in nonlinear models.

function cre(df::DataFrame, formula::FormulaTerm, id::Symbol; family = :probit, kwargs...)
    # 1. Identify covariates from formula
    formula_main, formula_fe = parse_fe(formula)
    
    # Get original covariates
    vars = StatsModels.termvars(formula_main.rhs)
    
    # 2. Compute group means for time-varying variables
    df_cre = copy(df)
    new_vars = Symbol[]
    for v in vars
        # Only add means for variables that vary within group
        # Fast check for within-group variation
        is_varying = let
            # Calculate within-group range or variance
            # combine(groupby(df, id), v => (x -> length(unique(x)) > 1) => :v).v |> any
            
            # More robust check: compare variable to its group mean
            means = combine(groupby(df, id), v => mean => :m)
            joined = leftjoin(df[!, [id, v]], means, on = id)
            any(joined[!, v] .!= joined.m)
        end
        
        if is_varying
            mean_var = Symbol("mean_", v)
            means = combine(groupby(df, id), v => mean => :m)
            # Ensure it is a Float64 vector to avoid being treated as categorical
            df_cre[!, mean_var] = convert(Vector{Float64}, leftjoin(df[!, [id]], means, on = id).m)
            push!(new_vars, mean_var)
        end
    end
    
    # 3. Construct new formula
    rhs_terms = [eachterm(formula_main.rhs)...]
    for nv in new_vars
        push!(rhs_terms, Term(nv))
    end
    
    new_formula = FormulaTerm(formula_main.lhs, tuple(rhs_terms...))
    
    return feglm(df_cre, new_formula; family = family, kwargs...)
end
