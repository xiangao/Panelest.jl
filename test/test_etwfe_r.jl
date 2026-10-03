# ETWFE against R etwfe 0.6.2 / marginaleffects. Fixtures and reference values
# are written by test/data/make_etwfe_ref.R.
using DelimitedFiles

function _read_csv(path)
    m, h = readdlm(path, ',', header = true)
    h = replace.(vec(h), "\"" => "")
    df = DataFrame()
    for (j, name) in enumerate(h)
        col = m[:, j]
        isna(x) = x isa AbstractString && x in ("NA", "\"NA\"")
        df[!, Symbol(name)] = all(x -> x isa Number || isna(x), col) ?
            [isna(x) ? NaN : Float64(x) for x in col] :
            String.(replace.(string.(col), "\"" => ""))
    end
    return df
end

@testset "ETWFE matches R etwfe" begin
    dir = joinpath(@__DIR__, "data")
    ref = _read_csv(joinpath(dir, "etwfe_ref.csv"))
    mp  = _read_csv(joinpath(dir, "etwfe_mpdta.csv"))
    st  = _read_csv(joinpath(dir, "etwfe_staggered.csv"))
    pc  = _read_csv(joinpath(dir, "etwfe_poisson.csv"))

    fits = Dict(
        "mpdta_lpop"       => etwfe(mp, @formula(lemp ~ lpop); gvar = :first_treat, tvar = :year,
                                    vcov = Vcov.cluster(:countyreal)),
        "mpdta_lpop_never" => etwfe(mp, @formula(lemp ~ lpop); gvar = :first_treat, tvar = :year,
                                    cgroup = "never", vcov = Vcov.cluster(:countyreal)),
        "mpdta_noctrl"     => etwfe(mp, @formula(lemp ~ 1); gvar = :first_treat, tvar = :year,
                                    vcov = Vcov.cluster(:countyreal)),
        "staggered_x"      => etwfe(st, @formula(y ~ x); gvar = :first_treat, tvar = :year,
                                    vcov = Vcov.cluster(:id)),
        "poisson_never"    => etwfe(pc, @formula(count ~ 1); gvar = :first_treat, tvar = :year,
                                    family = "poisson", cgroup = "never", vcov = Vcov.cluster(:id)),
        "poisson_z"        => etwfe(pc, @formula(count ~ z); gvar = :first_treat, tvar = :year,
                                    family = "poisson", vcov = Vcov.cluster(:id)),
    )

    for case in unique(ref.case), ty in unique(ref.type[ref.case .== case])
        r = ref[(ref.case .== case) .& (ref.type .== ty), :]
        e = ty == "event_pre" ? emfx(fits[case], type = "event", post_only = false) :
                                emfx(fits[case], type = ty)
        @testset "$case $ty" begin
            @test nrow(e) == nrow(r)
            if ty != "simple"
                @test e[!, ty == "calendar" ? :calendar : :event] == Int.(r.key)
            end
            @test isapprox(e.estimate, r.estimate; atol = 1e-6, rtol = 1e-6)
            rse = r.std_error
            @test all(isnan.(e.std_error) .== isnan.(rse))
            ok = .!isnan.(rse)
            @test isapprox(e.std_error[ok], rse[ok]; rtol = 1e-4)
        end
    end
end
