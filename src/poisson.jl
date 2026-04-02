function fepois(df::DataFrame, formula::FormulaTerm; vcov = Vcov.simple(), weights = nothing, kwargs...)
    # 1. Parse FE
    formula_main, formula_fe = parse_fe(formula)
    
    # 2. Get Fixed Effects
    fes, feids, fekeys = parse_fixedeffect(df, formula_fe)
    
    # 3. Model matrix for main part
    sch = schema(formula_main, df)
    f_schema = apply_schema(formula_main, sch, StatisticalModel)
    
    y = response(f_schema, df)
    X = modelmatrix(f_schema, df)
    coefnames_X = coefnames(f_schema)[2]
    if !isa(coefnames_X, Vector)
        coefnames_X = [coefnames_X]
    end
    
    # 4. Handle weights
    w_vec = weights === nothing ? ones(length(y)) : df[!, weights]

    # 5. Call IRLS
    res = irls_fit(y, X, fes; method = :poisson, weights = w_vec, kwargs...)
    
    # 6. VCov
    V = vcov_panelest(df, res, vcov; weights = w_vec)
    
    return PanelestModel(
        res.beta,
        V,
        res.residuals,
        res.mu,
        res.eta,
        res.converged,
        res.iterations,
        length(y),
        Int(sum(w_vec)) - size(X, 2) - length(fes),
        formula,
        coefnames_X,
        :poisson,
        res.X_resid,
        res.XtWX
    )
end
