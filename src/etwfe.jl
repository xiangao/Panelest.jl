# ---------------------------------------------------------------------------
# ETWFE post-processing: emfx and dataset
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

"""
    emfx(model; type="simple", post_only=true, gvar="first_treat_str",
          tvar="year_str", level=95.0)

Aggregate cohort×time ETWFE coefficients from a `feols` model into average
treatment effects.

- `type`: `"simple"` (overall ATT), `"event"` (event-study), or `"calendar"`
- `post_only`: restrict to post-treatment cells T ≥ G (default `true`)
- `gvar`, `tvar`: cohort and time column names used in the interaction term
"""
function emfx(model;
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
        return DataFrame(type=["ATT"], estimate=[τ], std_error=[se],
                         conf_low=[τ - z*se], conf_high=[τ + z*se])

    elseif type == "event"
        df = DataFrame(event=Int[], estimate=Float64[], std_error=Float64[],
                       conf_low=Float64[], conf_high=Float64[])
        for e in sort(unique(c.T - c.G for c in cells))
            τ, se = _aggregate(β, V, [c.idx for c in cells if c.T - c.G == e])
            push!(df, (e, τ, se, τ - z*se, τ + z*se))
        end
        return df

    elseif type == "calendar"
        df = DataFrame(calendar=Int[], estimate=Float64[], std_error=Float64[],
                       conf_low=Float64[], conf_high=Float64[])
        for t in sort(unique(c.T for c in cells))
            τ, se = _aggregate(β, V, [c.idx for c in cells if c.T == t])
            push!(df, (t, τ, se, τ - z*se, τ + z*se))
        end
        return df

    else
        error("emfx: unknown type '$type'. Choose \"simple\", \"event\", or \"calendar\".")
    end
end

"""
    dataset("mpdta")

Return a synthetic balanced panel mimicking Callaway & Sant'Anna's `mpdta`:
500 counties × 4 years (2004–2007), three treated cohorts (2004, 2006, 2007)
and a never-treated group. Includes `first_treat_str` and `year_str` string
columns required for ETWFE cohort×time interactions via StatsModels.
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
