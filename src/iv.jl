# Instrumental Variables / Two-Stage Least Squares (2SLS)

using LinearAlgebra, FixedEffects, Vcov, Distributions

# --- Structs ---

struct IVDiagnostics
    first_stage_F::Vector{Float64}
    wu_hausman::NamedTuple{(:stat, :p, :df1, :df2), NTuple{4, Float64}}
    sargan::Union{Nothing, NamedTuple{(:stat, :p, :df), NTuple{3, Float64}}}
end

struct IVModel <: StatisticalModel
    beta::Vector{Float64}
    vcov_mat::Matrix{Float64}
    residuals::Vector{Float64}
    nobs::Int
    df_residual::Int
    formula::FormulaTerm
    coefnames_vec::Vector{String}
    X_resid::Matrix{Float64}
    XtWX::Matrix{Float64}
    endo_names::Vector{Symbol}
    inst_names::Vector{Symbol}
    first_stages::Vector{PanelestModel}
    diagnostics::IVDiagnostics
end

# --- StatsAPI methods ---

StatsAPI.coef(m::IVModel) = m.beta
StatsAPI.vcov(m::IVModel) = m.vcov_mat
StatsAPI.stderror(m::IVModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.coefnames(m::IVModel) = m.coefnames_vec
StatsAPI.nobs(m::IVModel) = m.nobs
StatsAPI.dof_residual(m::IVModel) = m.df_residual
StatsAPI.residuals(m::IVModel) = m.residuals
StatsAPI.modelmatrix(m::IVModel) = m.X_resid
Vcov.invcrossmodelmatrix(m::IVModel) = pinv(m.XtWX)

function StatsBase.coeftable(m::IVModel)
    se = stderror(m)
    t_stats = m.beta ./ se
    p_vals = 2 * ccdf.(Normal(), abs.(t_stats))
    CoefTable(
        hcat(m.beta, se, t_stats, p_vals),
        ["Estimate", "Std. Error", "z value", "Pr(>|z|)"],
        m.coefnames_vec,
        4, 3
    )
end

function Base.show(io::IO, m::IVModel)
    println(io, "Panelest IV/2SLS Model")
    println(io, "Number of obs: ", m.nobs)
    println(io, "Endogenous:    ", join(string.(m.endo_names), ", "))
    println(io, "Instruments:   ", join(string.(m.inst_names), ", "))
    println(io)

    # Coefficient table
    ct = coeftable(m)
    show(io, ct)
    println(io)

    # Diagnostics
    d = m.diagnostics
    println(io, "\n--- Diagnostics ---")
    for (i, name) in enumerate(m.endo_names)
        println(io, "First-stage F ($name): $(round(d.first_stage_F[i], digits=2))")
    end
    println(io, "Wu-Hausman:   F($(Int(d.wu_hausman.df1)), $(Int(d.wu_hausman.df2))) = $(round(d.wu_hausman.stat, digits=4))  [p = $(round(d.wu_hausman.p, digits=4))]")
    if d.sargan !== nothing
        println(io, "Sargan:       χ²($(Int(d.sargan.df)))   = $(round(d.sargan.stat, digits=4))  [p = $(round(d.sargan.p, digits=4))]")
    end
end

# --- Helper: simple OLS on demeaned data ---

function _ols_demeaned(y::AbstractVector, X::AbstractMatrix)
    XtX = X' * X
    Xty = X' * y
    beta = pinv(XtX) * Xty
    resid = y .- X * beta
    return beta, resid, XtX
end

# --- Main feiv function ---

function feiv(df::DataFrame, formula::FormulaTerm;
              endo::Union{Symbol, Vector{Symbol}},
              inst::Union{Symbol, Vector{Symbol}},
              vcov_type = Vcov.simple(),
              weights = nothing,
              fixef_tol = 1e-8, fixef_maxiter = 10000)

    # Normalize to vectors
    endo_vec = endo isa Symbol ? [endo] : endo
    inst_vec = inst isa Symbol ? [inst] : inst
    n_endo = length(endo_vec)
    n_inst = length(inst_vec)

    # Order condition
    n_inst >= n_endo || throw(ArgumentError(
        "Under-identified: $n_inst instruments for $n_endo endogenous variables"))

    # --- Step 1: Parse formula and build matrices ---
    formula_main, formula_fe = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(df, formula_fe)
    n_fe = length(fes)

    sch = schema(formula_main, df)
    f_schema = apply_schema(formula_main, sch, StatisticalModel)

    y = Float64.(response(f_schema, df))
    X_exo = Float64.(modelmatrix(f_schema, df))
    exo_names = coefnames(f_schema)[2]
    exo_names = exo_names isa Vector ? exo_names : [exo_names]
    p_exo = size(X_exo, 2)

    X_endo = Float64.(hcat([df[!, c] for c in endo_vec]...))
    Z_inst = Float64.(hcat([df[!, c] for c in inst_vec]...))

    n = length(y)
    w_vec = weights === nothing ? ones(n) : Float64.(df[!, weights])

    # --- Step 2: FE absorption (demean all variables in one pass) ---
    has_fes = !isempty(fes)
    all_vars = hcat(y, X_exo, X_endo, Z_inst)

    if has_fes
        feM = AbstractFixedEffectSolver{Float64}(fes, Weights(w_vec), Val{:cpu})
        cols = collect(eachcol(all_vars))
        solve_residuals!(cols, feM; tol=fixef_tol, maxiter=fixef_maxiter)
        # solve_residuals! modifies in-place via the column views
    end

    y_dm = all_vars[:, 1]
    X_exo_dm = all_vars[:, 2:1+p_exo]
    X_endo_dm = all_vars[:, 2+p_exo:1+p_exo+n_endo]
    Z_inst_dm = all_vars[:, 2+p_exo+n_endo:end]

    # --- Step 3: First stage(s) ---
    ZX = hcat(Z_inst_dm, X_exo_dm)  # first-stage regressors
    p_zx = size(ZX, 2)

    X_endo_hat = similar(X_endo_dm)
    first_stage_F = Float64[]
    first_stages = PanelestModel[]

    for k in 1:n_endo
        u_k = X_endo_dm[:, k]

        # Unrestricted: regress endo on [Z_inst, X_exo]
        beta_fs, resid_fs, XtX_fs = _ols_demeaned(u_k, ZX)
        X_endo_hat[:, k] = ZX * beta_fs
        ssr_u = sum(resid_fs .^ 2)

        # Restricted: regress endo on X_exo only
        if p_exo > 0
            beta_r, resid_r, _ = _ols_demeaned(u_k, X_exo_dm)
            ssr_r = sum(resid_r .^ 2)
        else
            ssr_r = sum(u_k .^ 2)
        end

        # F-statistic
        df_num = n_inst
        df_denom = n - p_zx - n_fe
        F_k = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_denom)
        push!(first_stage_F, F_k)

        # Build PanelestModel for first stage (for storage/display)
        fs_XtWX = XtX_fs
        fs_V = pinv(fs_XtWX) * (ssr_u / df_denom)
        fs_coefnames = vcat(string.(inst_vec), exo_names)
        fs_model = PanelestModel(
            beta_fs, fs_V, resid_fs, ZX * beta_fs, ZX * beta_fs,
            true, 1, n, df_denom,
            formula, fs_coefnames, :iv_first_stage,
            ZX, fs_XtWX
        )
        push!(first_stages, fs_model)
    end

    # --- Step 4: Second stage ---
    UX = hcat(X_endo_hat, X_exo_dm)  # fitted endo + exo
    beta_2sls, resid_naive, XtWX_2sls = _ols_demeaned(y_dm, UX)

    # --- Step 5: Residual correction (use actual endo, not fitted) ---
    beta_endo = beta_2sls[1:n_endo]
    beta_exo = beta_2sls[n_endo+1:end]
    resid_corrected = y_dm .- X_endo_dm * beta_endo .- X_exo_dm * beta_exo

    # --- Step 6: Variance-covariance ---
    p_total = size(UX, 2)
    df_resid = n - p_total - n_fe

    # Build a result-like NamedTuple for vcov_panelest
    iv_res = (X_resid = UX, residuals = resid_corrected, XtWX = XtWX_2sls)
    V = vcov_panelest(df, iv_res, vcov_type; weights = w_vec)

    # --- Step 7: Diagnostics ---

    # Wu-Hausman test
    # Regress y on [first_stage_resids, X_endo_hat, X_exo]
    # Test whether first_stage_resid coefficients are jointly zero
    fs_resids = X_endo_dm .- X_endo_hat  # n × n_endo matrix of first-stage residuals
    RHS_wh = hcat(fs_resids, UX)
    beta_wh, resid_wh, _ = _ols_demeaned(y_dm, RHS_wh)
    ssr_wh = sum(resid_wh .^ 2)
    ssr_2sls = sum(resid_corrected .^ 2)

    wh_df1 = Float64(n_endo)
    wh_df2 = Float64(n - size(RHS_wh, 2) - n_fe)
    wh_stat = ((ssr_2sls - ssr_wh) / wh_df1) / (ssr_wh / wh_df2)
    wh_p = 1.0 - cdf(FDist(wh_df1, wh_df2), max(wh_stat, 0.0))
    wu_hausman = (stat = wh_stat, p = wh_p, df1 = wh_df1, df2 = wh_df2)

    # Sargan overidentification test (only when overidentified)
    sargan = nothing
    if n_inst > n_endo
        ZX_full = hcat(Z_inst_dm, X_exo_dm)
        _, resid_sargan, _ = _ols_demeaned(resid_corrected, ZX_full)
        ssr_sargan = sum(resid_sargan .^ 2)
        ssr_resid = sum(resid_corrected .^ 2)
        sargan_df = n_inst - n_endo
        sargan_stat = n * (1.0 - ssr_sargan / ssr_resid)
        sargan_p = 1.0 - cdf(Chisq(sargan_df), max(sargan_stat, 0.0))
        sargan = (stat = sargan_stat, p = sargan_p, df = Float64(sargan_df))
    end

    diagnostics = IVDiagnostics(first_stage_F, wu_hausman, sargan)

    # --- Build result ---
    all_coefnames = vcat(string.(endo_vec), exo_names)

    return IVModel(
        beta_2sls, V, resid_corrected,
        n, df_resid, formula, all_coefnames,
        UX, XtWX_2sls,
        endo_vec, inst_vec,
        first_stages, diagnostics
    )
end
