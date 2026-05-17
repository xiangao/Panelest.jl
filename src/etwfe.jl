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
    model     :: PanelestModel
    post_cols :: Vector{String}   # post-treatment dummy coefnames
    cohorts   :: Vector{Int}      # treated cohort values (gvar > 0)
    trange    :: UnitRange{Int}   # min:max of tvar
    gvar      :: Symbol
    tvar      :: Symbol
    family    :: String
end

# ---------------------------------------------------------------------------
# etwfe() — Wooldridge ETWFE with cohort FE + year FE (not unit FE)
# ---------------------------------------------------------------------------

"""
    etwfe(data, fml; gvar, tvar, family="gaussian", vcov=Vcov.simple(), kwargs...)

Fit Wooldridge's Extended Two-Way Fixed Effects (ETWFE) estimator.

Uses **cohort FE + year FE** (not unit FE) — the correct specification that
avoids contamination bias from mixing pre- and post-treatment variation in the
within-unit demeaning. Post-treatment cohort×year dummies are created
automatically for each (cohort g, year t ≥ g).

# Arguments
- `data`   : DataFrame
- `fml`    : outcome ~ controls formula (e.g. `@formula(y ~ 1)` or `@formula(y ~ x)`)
- `gvar`   : Symbol — integer cohort column (0 = never treated)
- `tvar`   : Symbol — integer time column
- `family` : `"gaussian"` (default, uses `feols`) or `"poisson"` (uses `fepois`)
- `vcov`   : variance estimator (default `Vcov.simple()`)

# Returns
An `ETWFEResult` that can be passed to `emfx()`.

# Example
```julia
mod = etwfe(df, @formula(count ~ 1); gvar=:first_treat, tvar=:year,
            family="poisson", vcov=Vcov.cluster(:id))
emfx(mod)
emfx(mod, type="event")
```
"""
function etwfe(data::DataFrame, fml::FormulaTerm;
               gvar   :: Symbol,
               tvar   :: Symbol,
               family :: String     = "gaussian",
               vcov                 = Vcov.simple(),
               kwargs...)

    cohorts = sort(unique(filter(g -> g > 0, data[!, gvar])))
    isempty(cohorts) && error("etwfe: no treated units found (gvar > 0)")
    tmin, tmax = extrema(data[!, tvar])

    df = copy(data)

    # Cohort and year FE as string columns (unique internal names)
    gfe_col = :_etwfe_g_str
    tfe_col = :_etwfe_t_str
    df[!, gfe_col] = string.(df[!, gvar])
    df[!, tfe_col] = string.(df[!, tvar])

    # Post-treatment dummies: one per (cohort g, post-treatment year t ≥ g)
    post_cols = String[]
    for g in cohorts, t in g:tmax
        col = "_D_g$(g)_t$(t)"
        df[!, Symbol(col)] = Float64.((df[!, gvar] .== g) .& (df[!, tvar] .== t))
        push!(post_cols, col)
    end

    # Build formula: controls + post dummies + cohort FE + year FE
    control_terms = [t for t in eachterm(fml.rhs) if !isa(t, ConstantTerm)]
    dummy_terms   = [StatsModels.term(Symbol(c)) for c in post_cols]
    all_rhs       = vcat(control_terms, dummy_terms,
                         AbstractTerm[fe(gfe_col), fe(tfe_col)])
    new_rhs       = length(all_rhs) == 1 ? all_rhs[1] : tuple(all_rhs...)
    new_fml       = FormulaTerm(fml.lhs, new_rhs)

    fitted = if lowercase(family) == "poisson"
        fepois(df, new_fml; vcov = vcov, kwargs...)
    else
        feols(df, new_fml; vcov = vcov, kwargs...)
    end

    return ETWFEResult(fitted, post_cols, cohorts, tmin:tmax, gvar, tvar, family)
end

# ---------------------------------------------------------------------------
# emfx() for ETWFEResult
# ---------------------------------------------------------------------------

"""
    emfx(result::ETWFEResult; type="simple", post_only=true, level=95.0)

Aggregate ETWFE treatment effects from an `ETWFEResult`.

- `type`: `"simple"` (overall ATT), `"event"` (by exposure time), `"calendar"` (by year)
- `post_only`: if `true` (default), restrict to post-treatment cells only
- `level`: confidence level (90, 95, or 99)

For Poisson models, estimates are on the **log scale** (log incidence rate ratios).
Exponentiate and subtract 1 for the proportional effect.
"""
function emfx(result::ETWFEResult;
              type      = "simple",
              post_only = true,
              level     = 95.0)

    model = result.model
    β = model.beta
    V = model.vcov
    z = _z_value(level)

    pat = r"^_D_g(\d+)_t(\d+)$"
    cells = NamedTuple{(:idx, :G, :T), Tuple{Int,Int,Int}}[]
    for col in result.post_cols
        m = match(pat, col)
        m === nothing && continue
        idx = findfirst(==(col), model.coefnames)
        idx === nothing && continue
        push!(cells, (idx = idx,
                      G   = parse(Int, m.captures[1]),
                      T   = parse(Int, m.captures[2])))
    end

    post_only && filter!(c -> c.T >= c.G, cells)
    isempty(cells) && error("emfx: no post-treatment coefficients found in model")

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
500 counties × 4 years (2004–2007), three treated cohorts (2004, 2006, 2007)
and a never-treated group. Includes `first_treat_str` and `year_str` string
columns required for the legacy ETWFE formula interface.
"""
function dataset(name::String)
    name == "mpdta" || error("Unknown dataset: '$name'")

    years        = [2004, 2005, 2006, 2007]
    n_per_cohort = 125
    cohorts      = vcat(fill(2004, n_per_cohort), fill(2006, n_per_cohort),
                        fill(2007, n_per_cohort), fill(0,    n_per_cohort))
    n            = length(cohorts)

    countyreal  = repeat(1:n,           inner = length(years))
    year_vec    = repeat(years,         outer = n)
    first_treat = repeat(cohorts,       inner = length(years))
    treat       = [g > 0 && t >= g ? 1 : 0 for (g, t) in zip(first_treat, year_vec)]

    unit_fe = repeat([sin(i * 0.37) for i in 1:n], inner = length(years))
    time_fe = repeat([0.0, 0.08, 0.16, 0.24],      outer = n)
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
