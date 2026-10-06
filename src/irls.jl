using LinearAlgebra, StatsFuns, FixedEffects, Vcov

function irls_fit(y::AbstractVector{<:Real}, X::AbstractMatrix{Float64}, fes::AbstractVector; 
                 method = :poisson, tol = 1e-8, maxiter = 100, 
                 fixef_tol = 1e-8, fixef_maxiter = 10000,
                 weights::AbstractVector{<:Real} = ones(length(y)))
    n = length(y)
    p = size(X, 2)
    has_fes = !isempty(fes)
    
    # 1. Initialization
    z_orig = zeros(n)
    if method == :poisson
        z_orig .= log.(max.(y, 0.1))
    elseif method == :logit || method == :clogit
        z_orig .= logit.((y .+ 0.5) ./ (1.0 .+ 1.0))
    elseif method == :probit
        z_orig .= norminvcdf.(clamp.((y .+ 0.5) ./ (1.0 .+ 1.0), 1e-5, 1-1e-5))
    else
        z_orig .= y
    end
    
    Z_init_X = hcat(z_orig, X)
    init_cols = collect(eachcol(Z_init_X))
    if has_fes
        solve_residuals_fixest!(init_cols, fes, weights; tol = 1e-4, maxiter = 100)
    end
    
    X_init_resid = hcat(init_cols[2:p+1]...)
    Z_init_resid = init_cols[1]
    # Use robust solver for initialization
    # XtWX weighted by frequency weights
    XtWX_init = zeros(p, p)
    XtWz_init = zeros(p)
    @inbounds for row in 1:n
        w = weights[row]
        zr = Z_init_resid[row]
        for col in 1:p
            xr_col = X_init_resid[row, col]
            XtWz_init[col] += xr_col * w * zr
            for row_p in 1:col
                val = X_init_resid[row, row_p] * w * xr_col
                XtWX_init[row_p, col] += val
                if row_p != col
                    XtWX_init[col, row_p] += val
                end
            end
        end
    end
    beta = pinv(XtWX_init) * XtWz_init
    
    # Correct initialization of eta using absorbed values
    # Z_init_resid is the residual after demeaning z_orig
    # FE_part = z_orig - Z_init_resid
    # Total explained = X*beta + FE_part
    eta = X * beta .+ (z_orig .- Z_init_resid)
    
    if method == :poisson
        mu = exp.(eta)
    elseif method == :logit || method == :clogit
        mu = logistic.(eta)
    elseif method == :probit
        eta .= clamp.(eta, -8.0, 8.0)
        mu = normcdf.(eta)
    else
        mu = eta
    end

    last_deviance = Inf
    converged = false
    iterations = 0

    Z_X = zeros(n, p + 1)
    Z_X[:, 2:p+1] .= X
    cols = collect(eachcol(Z_X))
    
    XtWX = zeros(p, p)
    XtWz = zeros(p)

    feM = nothing
    # Total weights for IRLS = frequency weights * IRLS working weights
    total_w = zeros(n)
    if has_fes
        if method == :ols
            total_w .= weights
        else
            total_w .= weights .* max.(mu, 1e-10)
        end
        feM = AbstractFixedEffectSolver{Float64}(fes, Weights(total_w), Val{:cpu})
    end
    w_val = zeros(n) # working weights

    for i in 1:maxiter
        iterations = i
        if method == :poisson
            w_val .= mu # working weights are mu
            @inbounds for j in 1:n
                # z = eta + (y - mu)/mu
                Z_X[j, 1] = eta[j] + (y[j] - mu[j]) / mu[j]
            end
        elseif method == :logit
            w_val .= mu .* (1.0 .- mu)
            @inbounds for j in 1:n
                w_val[j] = max(w_val[j], 1e-10)
                Z_X[j, 1] = eta[j] + (y[j] - mu[j]) / w_val[j]
            end
        elseif method == :probit
            @inbounds for j in 1:n
                pdf_val = normpdf(eta[j])
                cdf_val = mu[j]
                w_val[j] = max(pdf_val^2 / (cdf_val * (1.0 - cdf_val) + 1e-15), 1e-10)
                Z_X[j, 1] = eta[j] + (y[j] - cdf_val) / max(pdf_val, 1e-15)
            end
        else # OLS case
            w_val .= 1.0
            Z_X[:, 1] .= y
        end
        
        # Combine frequency weights with IRLS working weights
        total_w .= weights .* w_val
        
        if has_fes
            current_fixef_tol = i < 3 ? max(1e-4, fixef_tol * 100) : fixef_tol
            solve_residuals_fixest!(cols, fes, total_w; tol = current_fixef_tol, maxiter = fixef_maxiter)
        end
        
        z_resid = view(Z_X, :, 1)
        X_resid = view(Z_X, :, 2:p+1)
        
        fill!(XtWX, 0.0)
        fill!(XtWz, 0.0)
        @inbounds for row in 1:n
            weight = total_w[row]
            zr = z_resid[row]
            w_zr = weight * zr
            for col in 1:p
                xr_col = X_resid[row, col]
                XtWz[col] += xr_col * w_zr
                for row_p in 1:col
                    val = X_resid[row, row_p] * weight * xr_col
                    XtWX[row_p, col] += val
                    if row_p != col
                        XtWX[col, row_p] += val
                    end
                end
            end
        end
        
        # Use robust pinv for IRLS step
        delta = pinv(XtWX) * XtWz
        beta .= delta
        
        @inbounds for j in 1:n
            z_orig_val = 0.0
            if method == :poisson
                z_orig_val = eta[j] + (y[j] - mu[j]) / mu[j]
            elseif method == :logit
                z_orig_val = eta[j] + (y[j] - mu[j]) / w_val[j]
            elseif method == :probit
                pdf_val = normpdf(eta[j])
                cdf_val = mu[j]
                z_orig_val = eta[j] + (y[j] - cdf_val) / max(pdf_val, 1e-15)
            else
                z_orig_val = y[j]
            end
            
            projection = z_orig_val - z_resid[j]
            
            eta_update = 0.0
            for k in 1:p
                eta_update += X_resid[j, k] * delta[k]
            end
            
            eta[j] = projection + eta_update
            if method == :poisson
                mu[j] = exp(eta[j])
            elseif method == :logit
                mu[j] = logistic(eta[j])
            elseif method == :probit
                eta[j] = clamp(eta[j], -8.0, 8.0)
                mu[j] = normcdf(eta[j])
            else
                mu[j] = eta[j]
            end
        end
        
        current_deviance = 0.0
        if method == :poisson
            @inbounds for j in 1:n
                if y[j] > 0
                    current_deviance += weights[j] * 2.0 * (y[j] * log(y[j] / mu[j]) - (y[j] - mu[j]))
                else
                    current_deviance += weights[j] * 2.0 * mu[j]
                end
            end
        elseif method == :logit || method == :clogit
            @inbounds for j in 1:n
                if y[j] == 1
                    current_deviance -= 2.0 * weights[j] * log(mu[j] + 1e-15)
                else
                    current_deviance -= 2.0 * weights[j] * log(1.0 - mu[j] + 1e-15)
                end
            end
        elseif method == :probit
            @inbounds for j in 1:n
                # Binary deviance is same formula as logit but with probit mu
                if y[j] == 1
                    current_deviance -= 2.0 * weights[j] * log(mu[j] + 1e-15)
                else
                    current_deviance -= 2.0 * weights[j] * log(1.0 - mu[j] + 1e-15)
                end
            end
        else # OLS
            @inbounds for j in 1:n
                current_deviance += weights[j] * (y[j] - mu[j])^2
            end
        end
        
        if method != :ols && abs(current_deviance - last_deviance) / (0.1 + abs(current_deviance)) < tol
            converged = true
            break
        end
        if method == :ols
            converged = true
            break
        end
        last_deviance = current_deviance
        Z_X[:, 2:p+1] .= X
    end
    
    X_resid_final = copy(X)
    if has_fes
        cols_final = collect(eachcol(X_resid_final))
        solve_residuals_fixest!(cols_final, fes, total_w; tol = fixef_tol, maxiter = fixef_maxiter)
    end

    return (beta = beta, mu = mu, eta = eta, converged = converged, iterations = iterations, residuals = y .- mu, XtWX = XtWX, X_resid = X_resid_final)
end

# Dummy model for Vcov compatibility
struct PanelestDummyModel <: RegressionModel
    X_resid::Matrix{Float64}
    residuals::Vector{Float64}
    XtWX::Matrix{Float64}
    df_residual::Int
end
StatsAPI.modelmatrix(m::PanelestDummyModel) = m.X_resid
StatsAPI.residuals(m::PanelestDummyModel) = m.residuals
StatsAPI.dof_residual(m::PanelestDummyModel) = m.df_residual
Vcov.invcrossmodelmatrix(m::PanelestDummyModel) = pinv(m.XtWX)

# Number of parameters absorbed by the fixed effects, following fixest's default
# small-sample correction, ssc(fixef.K = "nested"): one for the absorbed intercept,
# plus levels - 1 for each fixed effect. A fixed effect nested in a cluster variable
# counts for nothing. This also matches Stata's xtreg, fe vce(cluster).
function fe_dof(fes, clusters = nothing)
    isempty(fes) && return 0
    k = 1
    for fe in fes
        nested = clusters !== nothing && any(c -> isnested(fe.refs, c.groups), clusters)
        k += nested ? 0 : fe.n - 1
    end
    return k
end

# true if every level of `inner` falls inside a single level of `outer`
function isnested(inner, outer)
    seen = Dict{eltype(inner), eltype(outer)}()
    for (a, b) in zip(inner, outer)
        get!(seen, a, b) == b || return false
    end
    return true
end

# `fes` are the absorbed fixed effects. `scale_simple = true` (OLS, IV) multiplies the
# simple covariance by the residual variance; for likelihood models (Poisson, logit,
# probit) the simple covariance is the inverse information and is left as is.
#
# Weights are fixest-style analytic weights. `res.residuals` and `res.X_resid` are
# unweighted (demeaned) residuals and regressors and `res.XtWX` is the weighted bread,
# so the score of observation i is weights[i] * X_resid[i, :] * residuals[i]. Degrees of
# freedom count observations, not the sum of the weights, as in fixest.
function vcov_panelest(df, res, vcov_method; weights = ones(size(res.X_resid, 1)),
                       fes = FixedEffect[], scale_simple = false)
    n, p = size(res.X_resid)
    if vcov_method isa Vcov.SimpleCovariance
        V = pinv(res.XtWX)
        scale_simple || return V
        sigma2 = sum(weights .* abs2.(res.residuals)) / (n - p - fe_dof(fes))
        return V .* sigma2
    end

    v = Vcov.materialize(df, vcov_method)
    clusters = v isa Vcov.ClusterCovariance ? values(v.clusters) : nothing
    # Vcov scales by (N - 1) / dof_residual (clustered) or N / dof_residual (robust)
    dof = n - p - fe_dof(fes, clusters)
    dummy = PanelestDummyModel(res.X_resid, weights .* res.residuals, res.XtWX, dof)
    return StatsAPI.vcov(dummy, v)
end
