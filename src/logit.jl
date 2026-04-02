function felogit(df::DataFrame, formula::FormulaTerm; vcov = Vcov.simple(), weights = nothing, kwargs...)
    formula_main, formula_fe = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(df, formula_fe)
    
    sch = schema(formula_main, df)
    f_schema = apply_schema(formula_main, sch, StatisticalModel)
    
    y = response(f_schema, df)
    X = modelmatrix(f_schema, df)
    coefnames_X = coefnames(f_schema)[2]
    if !isa(coefnames_X, Vector)
        coefnames_X = [coefnames_X]
    end

    w_vec = weights === nothing ? ones(length(y)) : df[!, weights]

    res = irls_fit(y, X, fes; method = :logit, weights = w_vec, kwargs...)
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
        :logit, res.X_resid, res.XtWX
    )
end
