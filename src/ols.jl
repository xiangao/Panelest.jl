function feols(df::DataFrame, formula::FormulaTerm; vcov = Vcov.simple(), weights = nothing, kwargs...)
    formula_main, formula_fe = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(df, formula_fe)

    has_fes = !isempty(fes)

    sch = schema(formula_main, df)
    f_schema = apply_schema(formula_main, sch, StatisticalModel)

    y = response(f_schema, df)
    X = modelmatrix(f_schema, df)
    coefnames_X = coefnames(f_schema)[2]
    if !isa(coefnames_X, Vector)
        coefnames_X = [coefnames_X]
    end

    n = length(y)
    p = size(X, 2)

    w_vec = weights === nothing ? ones(n) : df[!, weights]

    if has_fes
        # Optimized: Direct demeaning + Cholesky solve
        # Demean y and X together
        solver = DemeanSolver([fe.refs for fe in fes], n)
        yX = hcat(y, X)
        yX_cols = collect(eachcol(yX))
        solve_residuals_fixest!(yX_cols, fes, w_vec; tol=1e-8, maxiter=10000, solver=solver)
        
        y_demeaned = vec(yX_cols[1])
        X_demeaned = hcat(yX_cols[2:end]...)
        
        # Cholesky solve
        XtX = X_demeaned' * X_demeaned
        Xty = X_demeaned' * y_demeaned
        beta = try
            C = cholesky(Symmetric(XtX))
            C \ Xty
        catch
            pinv(XtX) * Xty
        end
        
        # Recover FE effects and compute residuals
        eta = X * beta
        residuals_raw = y_demeaned .- X_demeaned * beta
        fe_effects = y .- X * beta .- residuals_raw
        mu = X * beta .+ fe_effects
        residuals = y .- mu
        XtWX = XtX
        X_resid_final = X_demeaned
        converged = true
        iterations = 1
        df_residual = n - p - length(fes)
    else
        # Simple OLS
        XtX = X' * X
        Xty = X' * y
        beta = cholesky(Symmetric(XtX)) \ Xty
        eta = X * beta
        mu = eta
        residuals = y .- mu
        XtWX = XtX
        X_resid_final = X
        converged = true
        iterations = 1
        df_residual = n - p
    end

    V = vcov_panelest(df, (beta=beta, residuals=residuals, XtWX=XtWX, X_resid=X_resid_final), vcov; weights=w_vec)

    return PanelestModel(
        beta,
        V,
        residuals,
        mu,
        eta,
        converged,
        iterations,
        n,
        Int(df_residual),
        formula,
        coefnames_X,
        :ols, X_resid_final, XtWX
    )
end
