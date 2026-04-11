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

    # --- Optimization: Unique Cases Trick (Berge Algorithm) ---
    # Collapse data into unique FE combinations if many observations share the same FE IDs
    has_fes = !isempty(fes)
    
    # Use 2-FE optimized solver if applicable
    use_2fe_opt = length(fes) == 2 && get(kwargs, :use_2fe_opt, true)
    
    if has_fes && get(kwargs, :collapse, !use_2fe_opt)
        # Old collapse trick (good for 3+ FEs or when many covariates are identical)
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
            y_c = df_collapsed.y
            w_c = df_collapsed.w
            X_c = Matrix(df_collapsed[:, [Symbol("x$j") for j in 1:size(X, 2)]])
            fes_c = [FixedEffect(df_collapsed[!, Symbol("fe$i")]) for i in 1:length(fes)]
            
            res = fepois_fit(y_c, X_c, fes_c; weights = w_c, kwargs...)
            
            group_indices = groupindices(gdf)
            mu_full = res.mu[group_indices]
            eta_full = res.eta[group_indices]
            
            X_resid_full = copy(X)
            total_w_full = w_vec .* mu_full
            solve_residuals_fixest!(collect(eachcol(X_resid_full)), fes, total_w_full; tol=1e-8)
            
            V = vcov_panelest(df, (beta=res.beta, mu=mu_full, residuals=y .- mu_full, XtWX=res.XtWX, X_resid=X_resid_full), vcov; weights = w_vec)
            
            return PanelestModel(
                res.beta, V, y .- mu_full, mu_full, eta_full,
                res.converged, res.iterations, length(y),
                Int(sum(w_vec)) - size(X, 2) - length(fes),
                formula, coefnames_X, :poisson, X_resid_full, res.XtWX
            )
        end
    end

    # 5. Call concentrated likelihood solver (fixest style) - Standard path
    res = fepois_fit(y, X, fes; weights = w_vec, kwargs...)
    
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

function fepois_fit(y::AbstractVector{<:Real}, X::AbstractMatrix{Float64}, fes::Vector{FixedEffect};
                   weights::AbstractVector{Float64} = ones(length(y)),
                   tol = 1e-8, maxiter = 100, 
                   fixef_tol = 1e-8, fixef_maxiter = 10000, kwargs...)
    n = length(y)
    y_float = convert(Vector{Float64}, y)
    p = size(X, 2)
    has_fes = !isempty(fes)
    
    # Pre-allocate demeaning solver
    solver = has_fes ? DemeanSolver([fe.refs for fe in fes], n) : nothing

    # 1. Initialize beta (e.g. from OLS on log(y+0.1))
    z_init = log.(max.(y_float, 0.1))
    
    if has_fes
        # Initial demeaning of z and X for a good starting beta
        cols = collect(eachcol(hcat(z_init, X)))
        solve_residuals_fixest!(cols, fes, weights; tol=1e-4, solver=solver)
        z_init_resid = cols[1]
        X_init_resid = hcat(cols[2:end]...)
        beta = pinv(X_init_resid' * (weights .* X_init_resid)) * (X_init_resid' * (weights .* z_init_resid))
    else
        beta = pinv(X' * (weights .* X)) * (X' * (weights .* z_init))
    end
    
    # 2. Concentrated Maximum Likelihood Loop
    fes_refs = [convert(Vector{Int}, fe.refs) for fe in fes]
    fe_levels = [Int(maximum(fe.refs)) for fe in fes]
    
    # --- 2-FE Optimization (Unique Cells) ---
    fe2_solver = (length(fes) == 2 && get(kwargs, :use_2fe_opt, true)) ? 
                    TwoFEPoisSolver(y_float, weights, exp.(X * beta), fes_refs, fe_levels) : nothing
    
    eta = X * beta
    mu = exp.(eta)
    
    converged = false
    iterations = 0
    XtWX = zeros(p, p)
    X_resid = copy(X)

    for i in 1:maxiter
        iterations = i
        
        # A. For current beta, solve for optimal FE
        if has_fes
            if fe2_solver !== nothing
                exp_Xb = exp.(X * beta)
                update_mat_value!(fe2_solver, weights, exp_Xb)
                
                GX, fe_converged, fe_iters = fe_convergence_poisson_2(y_float, exp_Xb, fe2_solver; 
                                                                   tol = fixef_tol, maxiter = fixef_maxiter)
                # Recover mu from GX efficiently
                alpha_coef = view(GX, 1:fe_levels[1])
                beta_coef = view(GX, fe_levels[1]+1:fe_levels[1]+fe_levels[2])
                ref1 = fes_refs[1]
                ref2 = fes_refs[2]
                @inbounds for k in 1:n
                    mu[k] = exp_Xb[k] * alpha_coef[ref1[k]] * beta_coef[ref2[k]]
                end
            else
                mu, fe_converged, fe_iters = fe_convergence_poisson(
                    y_float, X * beta, fes_refs, fe_levels; 
                    weights = weights, tol = fixef_tol, maxiter = fixef_maxiter
                )
            end
        else
            mu .= exp.(X * beta)
        end
        
        # B. Compute gradient and Hessian of concentrated likelihood
        resid = y_float .- mu
        grad = X' * (weights .* resid)
        
        X_resid .= X
        if has_fes
            total_w = weights .* mu
            solve_residuals_fixest!(collect(eachcol(X_resid)), fes, total_w; tol=fixef_tol, solver=solver)
        end
        
        XtWX .= X_resid' * (weights .* mu .* X_resid)
        
        # C. Newton Step
        delta = try
            cholesky(Symmetric(XtWX)) \ grad
        catch
            pinv(XtWX) * grad
        end
        
        beta .+= delta
        
        # D. Check convergence on beta
        if maximum(abs.(delta)) < tol
            converged = true
            break
        end
    end
    
    # Final FE coefficients and mu
    if has_fes
        if fe2_solver !== nothing
            exp_Xb = exp.(X * beta)
            update_mat_value!(fe2_solver, weights, exp_Xb)
            GX, _, _ = fe_convergence_poisson_2(y_float, exp_Xb, fe2_solver; tol = fixef_tol, maxiter = fixef_maxiter)
            alpha_coef = view(GX, 1:fe_levels[1])
            beta_coef = view(GX, fe_levels[1]+1:fe_levels[1]+fe_levels[2])
            ref1 = fes_refs[1]
            ref2 = fes_refs[2]
            @inbounds for k in 1:n
                mu[k] = exp_Xb[k] * alpha_coef[ref1[k]] * beta_coef[ref2[k]]
            end
        else
            mu, _, _ = fe_convergence_poisson(
                y_float, X * beta, fes_refs, fe_levels; 
                weights = weights, tol = fixef_tol, maxiter = fixef_maxiter
            )
        end
    else
        mu .= exp.(X * beta)
    end
    
    return (beta = beta, mu = mu, eta = X * beta, converged = converged, iterations = iterations, residuals = y_float .- mu, XtWX = XtWX, X_resid = X_resid)
end
