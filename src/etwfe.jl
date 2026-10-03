# ---------------------------------------------------------------------------
# ETWFE: estimation, post-processing, and example data
# ---------------------------------------------------------------------------

function _z_value(level::Real)
    level ≈ 90.0 && return 1.64485
    level ≈ 95.0 && return 1.95996
    level ≈ 99.0 && return 2.57583
    return 1.95996
end

function _aggregate(β::Vector, V::Matrix, idxs::Vector{Int})
    n  = length(idxs)
    w  = fill(1.0 / n, n)
    τ  = dot(w, β[idxs])
    se = sqrt(max(0.0, w' * V[idxs, idxs] * w))
    return τ, se
end

# ---------------------------------------------------------------------------
# ETWFEResult — returned by etwfe(), accepted by emfx()
# ---------------------------------------------------------------------------

struct ETWFEResult
    model      :: PanelestModel
    post_cols  :: Vector{String}       # treatment-cell dummy coefnames, "_D_g{g}_t{t}"
    cohorts    :: Vector{Int}          # treated cohorts kept in the model
    trange     :: UnitRange{Int}       # min:max of tvar
    gvar       :: Symbol
    tvar       :: Symbol
    family     :: String
    cgroup     :: String               # "notyet" or "never"
    gref       :: Int                  # reference (control) cohort
    tref       :: Int                  # reference period
    controls   :: Vector{Symbol}
    # estimation sample, one entry per row used in the fit
    g          :: Vector{Int}
    t          :: Vector{Int}
    dtreat     :: BitVector            # Wooldridge's treatment-cell indicator
    cell_coef  :: Vector{Int}          # coef index of the row's cell dummy (0 = none)
    cellx_coef :: Matrix{Int}          # coef index of cell × control k (0 = none)
    xdm        :: Matrix{Float64}      # controls demeaned within cohort
    X          :: Matrix{Float64}      # full design (Poisson only; empty otherwise)
end

# ---------------------------------------------------------------------------
# etwfe() — Wooldridge ETWFE with cohort FE + year FE (not unit FE)
# ---------------------------------------------------------------------------

"""
    etwfe(data, fml; gvar, tvar, family="gaussian", cgroup="notyet",
          gref=nothing, tref=nothing, vcov=Vcov.simple(), kwargs...)

Fit Wooldridge's Extended Two-Way Fixed Effects (ETWFE) estimator, with the
same specification as R's `etwfe` (version 0.6).

The model has cohort and year effects (not unit effects) and one dummy for each
treatment cell, a (cohort g, period t) pair. With `cgroup = "notyet"` the cells
are the post-treatment periods t ≥ g, and units not yet treated serve as
controls. With `cgroup = "never"` every period except t = g - 1 gets a cell, so
only the never-treated serve as controls and the pre-treatment cells are free
(an event-study pre-trend check).

Each control x enters as in Wooldridge (2021): x itself, x interacted with the
cohort dummies and with the year dummies, and x demeaned within cohort
interacted with every treatment cell. The cell effects are then heterogeneous
in x, and `emfx()` averages them over the treated observations.

`family = "gaussian"` fits by `feols` with cohort and year fixed effects.
`family = "poisson"` fits by `fepois` with explicit cohort and year dummies, as R
`etwfe` does for nonlinear families, so that `emfx()` can compute count-scale
effects and their delta-method standard errors.

The reference cohort `gref` defaults to R's choice: a cohort coded after the
last period, else one coded before the first period (e.g. 0 for never treated),
else (`cgroup = "notyet"` only) the last-treated cohort, whose treated periods
are then dropped. Any other cohort treated at or before the first period has
no pre-treatment period; it is dropped with a warning (R would keep it, with
its cohort effect unidentified). The reference period `tref` defaults to the
first period.

# Arguments
- `data`   : DataFrame
- `fml`    : `outcome ~ controls`, e.g. `@formula(y ~ 1)` or `@formula(y ~ x1 + x2)`;
             controls must be plain numeric columns
- `gvar`   : integer cohort column (first treated period)
- `tvar`   : integer time column
- `family` : `"gaussian"` (default) or `"poisson"`
- `cgroup` : `"notyet"` (default) or `"never"`
- `vcov`   : variance estimator (default `Vcov.simple()`)

# Returns
An `ETWFEResult`, to pass to `emfx()`.

# Example
```julia
mod = etwfe(df, @formula(lemp ~ lpop); gvar=:first_treat, tvar=:year,
            vcov=Vcov.cluster(:countyreal))
emfx(mod)
emfx(mod, type="event")
```
"""
function etwfe(data::DataFrame, fml::FormulaTerm;
               gvar   :: Symbol,
               tvar   :: Symbol,
               family :: String     = "gaussian",
               cgroup :: String     = "notyet",
               gref                 = nothing,
               tref                 = nothing,
               vcov                 = Vcov.simple(),
               kwargs...)

    fam = lowercase(family)
    fam in ("gaussian", "poisson") || error("etwfe: family must be \"gaussian\" or \"poisson\"")
    cgroup in ("notyet", "never") || error("etwfe: cgroup must be \"notyet\" or \"never\"")

    gall = _as_int(data[!, gvar], gvar)
    tall = _as_int(data[!, tvar], tvar)
    ug = sort(unique(gall))
    ut = sort(unique(tall))
    tmin, tmax = first(ut), last(ut)

    # Reference cohort, following R etwfe
    gref_min_flag = false
    if gref === nothing
        cand = filter(>(tmax), ug)
        isempty(cand) && (cand = filter(<(tmin), ug))
        isempty(cand) && cgroup == "notyet" && (cand = [maximum(ug)])
        isempty(cand) && error("etwfe: the '$cgroup' control group could not be identified; pass `gref`.")
        gref = minimum(cand)
    else
        gref in ug || error("etwfe: gref = $gref not found in $gvar")
    end
    gref = Int(gref)
    gref < tmin && (gref_min_flag = true)
    tref = tref === nothing ? tmin : Int(tref)
    tref in ut || error("etwfe: tref = $tref not found in $tvar")

    cohorts_all = filter(!=(gref), ug)
    isempty(cohorts_all) && error("etwfe: no treated units found (every unit is in the reference cohort $gref)")

    # A cohort treated at or before the first sample period has no pre-treatment
    # observation, so its cohort effect is not separated from its treatment
    # cells, and it cannot serve as a control either. Drop it, the same way
    # Callaway & Sant'Anna (2021) drop always-treated units.
    dropped = filter(<=(tmin), cohorts_all)
    if !isempty(dropped)
        @warn "etwfe: dropping cohort(s) $(dropped) — treated at or before " *
              "the first sample period ($tmin), so no pre-treatment period is " *
              "observed. Their effect is not identified and they cannot serve " *
              "as a control (already treated throughout). Include earlier " *
              "periods in the data if you need to identify these cohorts."
    end
    cohorts = filter(>(tmin), cohorts_all)
    isempty(cohorts) && error(
        "etwfe: no identified cohorts remain — every treated cohort is " *
        "treated at or before the first sample period ($tmin).")

    # Controls: plain numeric columns only
    controls = Symbol[]
    for term in eachterm(fml.rhs)
        term isa ConstantTerm && continue
        term isa Term || error("etwfe: controls must be plain column names; got $(term)")
        push!(controls, term.sym)
    end
    K = length(controls)

    # Wooldridge's demeaned controls: x minus its cohort mean, computed on the
    # full data as R etwfe does (before any rows are dropped below)
    keep_rows = .!(in(dropped).(gall))
    xall = Matrix{Float64}(undef, length(gall), K)
    for (k, c) in enumerate(controls)
        xall[:, k] = Float64.(data[!, c])
    end
    xdm_all = similar(xall)
    for g in ug
        rows = findall(==(g), gall)
        for k in 1:K
            m = sum(@view xall[rows, k]) / length(rows)
            xdm_all[rows, k] .= xall[rows, k] .- m
        end
    end

    # Treatment-cell indicator
    if cgroup == "notyet"
        dtreat_all = (tall .>= gall) .& (gall .!= gref)
        # R sets .Dtreat to NA (row dropped) once the reference cohort is treated
        if !gref_min_flag
            keep_rows .&= tall .< gref
        end
    else
        for g in cohorts
            (g - 1) in ut || error("etwfe: cgroup = \"never\" needs period $(g - 1), " *
                                   "the one before cohort $g is treated")
        end
        dtreat_all = tall .!= gall .- 1
    end

    rows = findall(keep_rows)
    g = gall[rows]
    t = tall[rows]
    dtreat = dtreat_all[rows]
    xdm = xdm_all[rows, :]
    xs = xall[rows, :]
    n = length(rows)

    df = DataFrame()
    yname = fml.lhs.sym
    df[!, yname] = data[rows, yname]
    cluster_cols = vcov isa Vcov.ClusterCovariance ? collect(vcov.clusternames) : Symbol[]
    for c in cluster_cols
        c in propertynames(df) || (df[!, c] = data[rows, c])
    end

    # Treatment cells that have observations
    cells = Tuple{Int,Int}[]
    for gg in cohorts, tt in ut
        cgroup == "notyet" && tt < gg && continue
        cgroup == "never" && tt == gg - 1 && continue
        any((g .== gg) .& (t .== tt)) && push!(cells, (gg, tt))
    end

    rhs_cols = String[]
    post_cols = String[]
    cellx_cols = Matrix{String}(undef, length(cells), K)
    for (j, (gg, tt)) in enumerate(cells)
        d = Float64.((g .== gg) .& (t .== tt))
        col = "_D_g$(gg)_t$(tt)"
        df[!, col] = d
        push!(post_cols, col); push!(rhs_cols, col)
        for (k, c) in enumerate(controls)
            cx = "$(col)_x_$(c)"
            df[!, cx] = d .* xdm[:, k]
            cellx_cols[j, k] = cx
            push!(rhs_cols, cx)
        end
    end
    for (k, c) in enumerate(controls)
        df[!, "_x_$(c)"] = xs[:, k]
        push!(rhs_cols, "_x_$(c)")
        for gg in cohorts
            col = "_x_$(c)_g$(gg)"
            df[!, col] = xs[:, k] .* (g .== gg)
            push!(rhs_cols, col)
        end
        for tt in ut
            tt == tref && continue
            col = "_x_$(c)_t$(tt)"
            df[!, col] = xs[:, k] .* (t .== tt)
            push!(rhs_cols, col)
        end
    end

    terms = AbstractTerm[StatsModels.term(Symbol(c)) for c in rhs_cols]
    if fam == "gaussian"
        df[!, :_etwfe_g] = g
        df[!, :_etwfe_t] = t
        push!(terms, fe(:_etwfe_g), fe(:_etwfe_t))
        fitted = feols(df, FormulaTerm(fml.lhs, tuple(terms...)); vcov = vcov, kwargs...)
    else
        # R etwfe uses fe = "none" for nonlinear families: explicit dummies
        for gg in cohorts
            col = "_G_$(gg)"
            df[!, col] = Float64.(g .== gg)
            push!(terms, StatsModels.term(Symbol(col)))
        end
        for tt in ut
            tt == tref && continue
            col = "_T_$(tt)"
            df[!, col] = Float64.(t .== tt)
            push!(terms, StatsModels.term(Symbol(col)))
        end
        fitted = fepois(df, FormulaTerm(fml.lhs, tuple(ConstantTerm(1), terms...)); vcov = vcov, kwargs...)
    end

    cn = fitted.coefnames
    cellpos = Dict(c => j for (j, c) in enumerate(cells))
    cell_coef = zeros(Int, n)
    cellx_coef = zeros(Int, n, K)
    for i in 1:n
        dtreat[i] || continue
        j = get(cellpos, (g[i], t[i]), 0)
        j == 0 && continue
        cell_coef[i] = something(findfirst(==(post_cols[j]), cn), 0)
        for k in 1:K
            cellx_coef[i, k] = something(findfirst(==(cellx_cols[j, k]), cn), 0)
        end
    end

    X = fam == "poisson" ? _design_matrix(df, cn) : Matrix{Float64}(undef, 0, 0)

    return ETWFEResult(fitted, post_cols, cohorts, tmin:tmax, gvar, tvar, fam,
                       cgroup, gref, tref, controls, g, t, BitVector(dtreat),
                       cell_coef, cellx_coef, xdm, X)
end

function _as_int(v, name)
    all(x -> !ismissing(x) && isinteger(x), v) ||
        error("etwfe: $name must be integer-valued with no missing values")
    return Int.(v)
end

function _design_matrix(df, cn)
    X = Matrix{Float64}(undef, nrow(df), length(cn))
    for (j, c) in enumerate(cn)
        X[:, j] = c == "(Intercept)" ? ones(nrow(df)) : df[!, c]
    end
    return X
end

# ---------------------------------------------------------------------------
# emfx() for ETWFEResult
# ---------------------------------------------------------------------------

"""
    emfx(result::ETWFEResult; type="simple", post_only=true, scale="response", level=95.0)

Average the ETWFE treatment effects over treated observations, as R's
`etwfe::emfx()` does.

For each treated observation the effect is the change in the prediction when
its treatment cell is switched on: the cell coefficient plus the cell × control
slopes times the demeaned controls. For Poisson models on the default
`scale = "response"`, the effect is exp(η) - exp(η - cell term), in count
units; `scale = "link"` gives the average log-scale effect instead. Standard
errors come from the delta method.

- `type`: `"simple"` (overall ATT), `"group"` (by cohort), `"event"` (by time
  since treatment) or `"calendar"` (by period)
- `post_only`: with `cgroup = "notyet"` and `type = "event"`, `false` adds the
  pre-treatment event times, which are zero by construction (SE reported as
  NaN). With `cgroup = "never"` the event aggregation always includes them, as
  in R.
- `level`: confidence level (90, 95, or 99)
"""
function emfx(result::ETWFEResult;
              type      = "simple",
              post_only = true,
              scale     = "response",
              level     = 95.0)

    type in ("simple", "group", "event", "calendar") ||
        error("emfx: unknown type '$type'. Choose \"simple\", \"group\", \"event\", or \"calendar\".")
    scale in ("response", "link") || error("emfx: scale must be \"response\" or \"link\"")

    r = result
    β = r.model.beta
    V = r.model.vcov
    z = _z_value(level)
    treated = r.g .!= r.gref

    # Rows to average over, as R emfx selects them
    use = if r.cgroup == "never"
        type == "event" ? treated : treated .& r.dtreat .& (r.t .>= r.g)
    elseif type == "event" && !post_only
        treated
    else
        r.dtreat
    end
    rows = findall(use)
    isempty(rows) && error("emfx: no treated observations")

    key = type == "simple"   ? zeros(Int, length(rows)) :
          type == "group"    ? r.g[rows] :
          type == "calendar" ? r.t[rows] :
                               r.t[rows] .- r.g[rows]

    response = r.family == "poisson" && scale == "response"
    p = length(β)
    keys_sorted = sort(unique(key))
    est = Float64[]; se = Float64[]
    for kv in keys_sorted
        sel = rows[key .== kv]
        a = zeros(p)
        τ = 0.0
        for i in sel
            ∂δ = zeros(p)
            δ = 0.0
            if r.cell_coef[i] > 0
                ∂δ[r.cell_coef[i]] = 1.0
                δ += β[r.cell_coef[i]]
                for k in eachindex(r.controls)
                    jx = r.cellx_coef[i, k]
                    jx > 0 || continue
                    ∂δ[jx] = r.xdm[i, k]
                    δ += β[jx] * r.xdm[i, k]
                end
            end
            if response
                xi = @view r.X[i, :]
                η = dot(xi, β)
                μ1, μ0 = exp(η), exp(η - δ)
                τ += μ1 - μ0
                a .+= μ1 .* xi .- μ0 .* (xi .- ∂δ)
            else
                τ += δ
                a .+= ∂δ
            end
        end
        m = length(sel)
        τ /= m
        a ./= m
        s = all(iszero, a) ? NaN : sqrt(max(0.0, a' * V * a))
        push!(est, τ); push!(se, s)
    end

    out = DataFrame()
    if type == "simple"
        out.type = ["ATT"]
    else
        out[!, Symbol(type)] = keys_sorted
    end
    out.estimate  = est
    out.std_error = se
    out.conf_low  = est .- z .* se
    out.conf_high = est .+ z .* se
    return out
end

# ---------------------------------------------------------------------------
# Legacy emfx for PanelestModel (string-pattern approach)
# ---------------------------------------------------------------------------

"""
    emfx(model::PanelestModel; type, post_only, gvar, tvar, level)

Aggregate cohort×time ETWFE coefficients from a manually-constructed `feols`
or `fepois` model. Looks for coefnames matching `"gvar: X & tvar: Y"`.

Prefer `etwfe()` + `emfx(::ETWFEResult)` for new code.
"""
function emfx(model::PanelestModel;
              type      = "simple",
              post_only = true,
              gvar      = "first_treat_str",
              tvar      = "year_str",
              level     = 95.0)

    names = model.coefnames
    β     = model.beta
    V     = model.vcov
    z     = _z_value(level)

    pat1 = Regex("^" * gvar * ": (\\S+) & " * tvar * ": (\\S+)\$")
    pat2 = Regex("^" * tvar * ": (\\S+) & " * gvar * ": (\\S+)\$")

    cells = NamedTuple{(:idx, :G, :T), Tuple{Int,Int,Int}}[]
    for (i, n) in enumerate(names)
        m = match(pat1, n)
        if !isnothing(m)
            push!(cells, (idx = i, G = parse(Int, m.captures[1]), T = parse(Int, m.captures[2])))
            continue
        end
        m = match(pat2, n)
        if !isnothing(m)
            push!(cells, (idx = i, G = parse(Int, m.captures[2]), T = parse(Int, m.captures[1])))
        end
    end

    filter!(c -> c.G > 0, cells)
    post_only && filter!(c -> c.T >= c.G, cells)

    isempty(cells) && error(
        "emfx: no cohort×time interaction coefficients found.\n" *
        "Expected coefnames like '$gvar: 2006 & $tvar: 2007'.\n" *
        "Check that gvar='$gvar' and tvar='$tvar' match the column names in your formula.")

    if type == "simple"
        τ, se = _aggregate(β, V, [c.idx for c in cells])
        return DataFrame(type      = ["ATT"],
                         estimate  = [τ],
                         std_error = [se],
                         conf_low  = [τ - z * se],
                         conf_high = [τ + z * se])

    elseif type == "event"
        df = DataFrame(event     = Int[],
                       estimate  = Float64[],
                       std_error = Float64[],
                       conf_low  = Float64[],
                       conf_high = Float64[])
        for e in sort(unique(c.T - c.G for c in cells))
            τ, se = _aggregate(β, V, [c.idx for c in cells if c.T - c.G == e])
            push!(df, (e, τ, se, τ - z * se, τ + z * se))
        end
        return df

    elseif type == "calendar"
        df = DataFrame(calendar  = Int[],
                       estimate  = Float64[],
                       std_error = Float64[],
                       conf_low  = Float64[],
                       conf_high = Float64[])
        for t in sort(unique(c.T for c in cells))
            τ, se = _aggregate(β, V, [c.idx for c in cells if c.T == t])
            push!(df, (t, τ, se, τ - z * se, τ + z * se))
        end
        return df

    else
        error("emfx: unknown type '$type'. Choose \"simple\", \"event\", or \"calendar\".")
    end
end

# ---------------------------------------------------------------------------
# dataset()
# ---------------------------------------------------------------------------

"""
    dataset("mpdta")

Return a synthetic balanced panel mimicking Callaway & Sant'Anna's `mpdta`:
500 counties × 5 years (2003–2007), three treated cohorts (2004, 2006, 2007)
and a never-treated group. Includes `first_treat_str` and `year_str` string
columns required for the legacy ETWFE formula interface.

The sample starts one year (2003) before the earliest treated cohort (2004),
matching the real `mpdta`, so every cohort has a genuine pre-treatment period
and its effect is identified by `etwfe()`.
"""
function dataset(name::String)
    name == "mpdta" || error("Unknown dataset: '$name'")

    years        = [2003, 2004, 2005, 2006, 2007]
    n_per_cohort = 125
    cohorts      = vcat(fill(2004, n_per_cohort), fill(2006, n_per_cohort),
                        fill(2007, n_per_cohort), fill(0,    n_per_cohort))
    n            = length(cohorts)

    countyreal  = repeat(1:n,           inner = length(years))
    year_vec    = repeat(years,         outer = n)
    first_treat = repeat(cohorts,       inner = length(years))
    treat       = [g > 0 && t >= g ? 1 : 0 for (g, t) in zip(first_treat, year_vec)]

    unit_fe = repeat([sin(i * 0.37) for i in 1:n], inner = length(years))
    time_fe = repeat([0.0, 0.08, 0.16, 0.24, 0.32], outer = n)
    eps     = [cos(i * 1.37) * 0.4               for i in 1:(n * length(years))]
    lemp    = unit_fe .+ time_fe .+ 0.30 .* treat .+ eps
    lpop    = repeat([10.0 + sin(i * 0.53) for i in 1:n], inner = length(years))

    return DataFrame(
        countyreal      = countyreal,
        year            = year_vec,
        year_str        = string.(year_vec),
        first_treat     = first_treat,
        first_treat_str = string.(first_treat),
        treat           = treat,
        lemp            = lemp,
        lpop            = lpop,
    )
end
