function feprobit(df::DataFrame, formula::FormulaTerm; vcov = Vcov.simple(), weights = nothing, kwargs...)
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

    # --- Optimization: Unique Cases Trick (Berge Algorithm) ---
    has_fes = !isempty(fes)
    if has_fes && get(kwargs, :collapse, true)
        df_tmp = DataFrame(y=y, w=w_vec)
        for (i, fe) in enumerate(fes)
            df_tmp[!, Symbol("fe$i")] = fe.refs
        end
        for j in 1:size(X, 2)
            df_tmp[!, Symbol("x$j")] = X[:, j]
        end
        
        gdf = groupby(df_tmp, [Symbol("fe$i") for i in 1:length(fes)] ∪ [Symbol("x$j") for j in 1:size(X, 2)])
        
        if length(gdf) < 0.8 * length(y)
            df_collapsed = combine(gdf, :y => sum => :y, :w => sum => :w, ungroup=true)
            # For probit, y_mean should be sum(y)/sum(w)
            y_c = df_collapsed.y ./ df_collapsed.w
            w_c = df_collapsed.w
            X_c = Matrix(df_collapsed[:, [Symbol("x$j") for j in 1:size(X, 2)]])
            fes_c = [FixedEffect(df_collapsed[!, Symbol("fe$i")]) for i in 1:length(fes)]
            
            res = feprobit_fit(y_c, X_c, fes_c; weights = w_c, kwargs...)
            
            group_indices = groupindices(gdf)
            mu_full = res.mu[group_indices]
            eta_full = res.eta[group_indices]
            
            X_resid_full = copy(X)
            # Weights for Probit Hessian are w * pdf^2 / (cdf * (1-cdf))
            pdf_vals_full = normpdf.(eta_full)
            total_w_full = w_vec .* pdf_vals_full.^2 ./ (max.(mu_full .* (1.0 .- mu_full), 1e-15))
            solve_residuals_fixest!(collect(eachcol(X_resid_full)), fes, total_w_full; tol=1e-8)
            
            V = vcov_panelest(df, (beta=res.beta, mu=mu_full, residuals=y .- mu_full, XtWX=res.XtWX, X_resid=X_resid_full), vcov; weights = w_vec)
            
            return PanelestModel(
                res.beta, V, y .- mu_full, mu_full, eta_full,
                res.converged, res.iterations, length(y),
                Int(sum(w_vec)) - size(X, 2) - length(fes),
                formula, coefnames_X, :probit, X_resid_full, res.XtWX
            )
        end
    end

    # 5. Call concentrated likelihood solver
    res = feprobit_fit(y, X, fes; weights = w_vec, kwargs...)
    
    # 6. VCov
    V = vcov_panelest(df, res, vcov; weights = w_vec)
    
    return PanelestModel(
        res.beta, V, res.residuals, res.mu, res.eta,
        res.converged, res.iterations, length(y),
        Int(sum(w_vec)) - size(X, 2) - length(fes),
        formula, coefnames_X, :probit, res.X_resid, res.XtWX
    )
end

function feprobit_fit(y::AbstractVector{<:Real}, X::AbstractMatrix{Float64}, fes::Vector{FixedEffect};
                     weights::AbstractVector{Float64} = ones(length(y)),
                     tol = 1e-8, maxiter = 100, 
                     fixef_tol = 1e-8, fixef_maxiter = 10000)
    n = length(y)
    y_float = convert(Vector{Float64}, y)
    p = size(X, 2)
    has_fes = !isempty(fes)
    
    solver = has_fes ? DemeanSolver([fe.refs for fe in fes], n) : nothing

    # 1. Initialize beta
    beta = zeros(p)
    
    # 2. Concentrated Maximum Likelihood Loop
    fes_refs = [convert(Vector{Int}, fe.refs) for fe in fes]
    fe_levels = [Int(maximum(fe.refs)) for fe in fes]
    
    eta = X * beta
    eta .= clamp.(eta, -8.0, 8.0)
    mu = normcdf.(eta)
    
    converged = false
    iterations = 0
    XtWX = zeros(p, p)
    X_resid = copy(X)

    for i in 1:maxiter
        iterations = i
        
        # A. For current beta, solve for optimal FE
        if has_fes
            mu, fe_converged, fe_iters = fe_convergence_probit(
                y_float, X * beta, fes_refs, fe_levels; 
                weights = weights, tol = fixef_tol, maxiter = fixef_maxiter
            )
        else
            mu .= normcdf.(clamp.(X * beta, -8.0, 8.0))
        end
        
        # B. Compute gradient and Hessian
        # Probit Score: w * (y - cdf) * pdf / (cdf * (1 - cdf))
        pdf_vals = normpdf.(norminvcdf.(mu)) # Since mu = cdf(eta + alpha)
        # Actually it's easier to just track eta
        # Wait, if we use the result from fe_convergence_probit, we have mu.
        # We need the eta that corresponds to this mu.
        eta_with_fe = norminvcdf.(clamp.(mu, 1e-15, 1.0 - 1e-15))
        pdf_vals = normpdf.(eta_with_fe)
        
        denom = max.(mu .* (1.0 .- mu), 1e-15)
        resid_score = (y_float .- mu) .* pdf_vals ./ denom
        grad = X' * (weights .* resid_score)
        
        X_resid .= X
        if has_fes
            # Expected Hessian weights for Probit: w * pdf^2 / (cdf * (1 - cdf))
            total_w = weights .* pdf_vals.^2 ./ denom
            solve_residuals_fixest!(collect(eachcol(X_resid)), fes, total_w; tol=fixef_tol, solver=solver)
        end
        
        XtWX .= X_resid' * (weights .* (pdf_vals.^2 ./ denom) .* X_resid)
        
        # C. Newton Step
        delta = try
            cholesky(Symmetric(XtWX)) \ grad
        catch
            pinv(XtWX) * grad
        end
        
        beta .+= delta
        
        # D. Check convergence
        if maximum(abs.(delta)) < tol
            converged = true
            break
        end
    end
    
    if has_fes
        mu, _, _ = fe_convergence_probit(
            y_float, X * beta, fes_refs, fe_levels; 
            weights = weights, tol = fixef_tol, maxiter = fixef_maxiter
        )
    else
        mu .= normcdf.(clamp.(X * beta, -8.0, 8.0))
    end
    
    return (beta = beta, mu = mu, eta = X * beta, converged = converged, iterations = iterations, residuals = y_float .- mu, XtWX = XtWX, X_resid = X_resid)
end
