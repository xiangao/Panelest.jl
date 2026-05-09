module Panelest

using DataFrames
using FixedEffects
using LinearAlgebra
using Statistics
using StatsAPI
using StatsBase
using StatsFuns
using StatsModels
using Vcov
using Tables
using Distributions

export fepois, felogit, feprobit, feols, feglm, fe, clogit, cre, feiv
export emfx, dataset

# --- Formula Parsing ---

struct FixedEffectTerm <: AbstractTerm
    x::Symbol
end
StatsModels.termvars(t::FixedEffectTerm) = [t.x]
fe(x::Term) = fe(Symbol(x))
fe(s::Symbol) = FixedEffectTerm(s)

has_fe(::FixedEffectTerm) = true
has_fe(::FunctionTerm{typeof(fe)}) = true
has_fe(@nospecialize(t::InteractionTerm)) = any(has_fe(x) for x in t.terms)
has_fe(::AbstractTerm) = false
has_fe(@nospecialize(t::FormulaTerm)) = any(has_fe(x) for x in eachterm(t.rhs))

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x

function parse_fe(@nospecialize(f::FormulaTerm))
    if has_fe(f)
        # Use StatsModels eachterm to handle interactions and single terms
        rhs_terms = [t for t in eachterm(f.rhs)]
        
        main_terms = AbstractTerm[term for term in rhs_terms if !has_fe(term)]
        fe_terms = AbstractTerm[term for term in rhs_terms if has_fe(term)]
        
        # Suppress intercept in formula_main by adding ConstantTerm(0)
        # only if not already suppressed and we HAVE fixed effects.
        if !any(isa(t, ConstantTerm) && t.n == 0 for t in main_terms)
            push!(main_terms, ConstantTerm(0))
        end
        
        # Reconstruct RHS
        rhs_main = length(main_terms) == 1 ? main_terms[1] : tuple(main_terms...)
        rhs_fe = length(fe_terms) == 1 ? fe_terms[1] : tuple(fe_terms...)
        
        formula_main = FormulaTerm(f.lhs, rhs_main)
        formula_fe = FormulaTerm(ConstantTerm(0), rhs_fe)
        return formula_main, formula_fe
    else
        return f, FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    end
end

fesymbol(t::FixedEffectTerm) = t.x
fesymbol(t::FunctionTerm{typeof(fe)}) = Symbol(t.args[1])

function parse_fixedeffect(data, @nospecialize(formula::FormulaTerm))
    fes = FixedEffect[]
    feids = Symbol[]
    fekeys = Symbol[]
    for term in eachterm(formula.rhs)
        result = _parse_fixedeffect(data, term)
        if result !== nothing
            push!(fes, result[1])
            push!(feids, result[2])
            append!(fekeys, result[3])
        end
    end
    return fes, feids, unique(fekeys)
end

function _parse_fixedeffect(data, @nospecialize(t::AbstractTerm))
    if has_fe(t)
        st = fesymbol(t)
        return FixedEffect(Tables.getcolumn(data, st)), Symbol(:fe_, st), [st]
    end
    return nothing
end

function _parse_fixedeffect(data, @nospecialize(t::InteractionTerm))
    fes_terms = (x for x in t.terms if has_fe(x))
    interactions = (x for x in t.terms if !has_fe(x))
    if !isempty(fes_terms)
        fe_names = [fesymbol(x) for x in fes_terms]
        v1 = _multiply(data, Symbol.(interactions))
        fe_obj = FixedEffect((Tables.getcolumn(data, fe_name) for fe_name in fe_names)...; interaction = v1)
        # interactions_str = string.(interactions)
        return fe_obj, Symbol("fe_interaction"), fe_names
    end
    return nothing
end

function _multiply(data, ss::AbstractVector)
    if isempty(ss)
        return uweights(size(data, 1))
    elseif length(ss) == 1
        return convert(AbstractVector{Float64}, replace(Tables.getcolumn(data, ss[1]), missing => 0))
    else
        return convert(AbstractVector{Float64}, replace!(.*((Tables.getcolumn(data, x) for x in ss)...), missing => 0))
    end
end

# --- Model Result Type ---

struct PanelestModel <: StatisticalModel
    beta::Vector{Float64}
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    mu::Vector{Float64}
    eta::Vector{Float64}
    converged::Bool
    iterations::Int
    nobs::Int
    df_residual::Int
    formula::FormulaTerm
    coefnames::Vector{String}
    method::Symbol # :poisson, :logit, :ols
    X_resid::Matrix{Float64}
    XtWX::Matrix{Float64}
end

StatsAPI.coef(m::PanelestModel) = m.beta
StatsAPI.vcov(m::PanelestModel) = m.vcov
StatsAPI.stderror(m::PanelestModel) = sqrt.(diag(m.vcov))
StatsAPI.coefnames(m::PanelestModel) = m.coefnames
StatsAPI.nobs(m::PanelestModel) = m.nobs
StatsAPI.dof_residual(m::PanelestModel) = m.df_residual
StatsAPI.modelmatrix(m::PanelestModel) = m.X_resid
StatsAPI.residuals(m::PanelestModel) = m.residuals

# Vcov compatibility
Vcov.invcrossmodelmatrix(m::PanelestModel) = pinv(m.XtWX)

function Base.show(io::IO, m::PanelestModel)
    println(io, "Panelest Model: ", m.method)
    println(io, "Number of obs: ", m.nobs)
    println(io, "Converged: ", m.converged)
    println(io, "Iterations: ", m.iterations)
    # Print coeftable
    ct = coeftable(m)
    show(io, ct)
end

function StatsBase.coeftable(m::PanelestModel)
    std_err = stderror(m)
    t_stats = m.beta ./ std_err
    p_vals = 2 * ccdf.(Normal(), abs.(t_stats))
    CoefTable(
        hcat(m.beta, std_err, t_stats, p_vals),
        ["Estimate", "Std. Error", "z value", "Pr(>|z|)"],
        m.coefnames,
        4, # p-value column
        3  # t-statistic column
    )
end

# --- Core IRLS Engine ---

include("direct_fe.jl")
include("fe_convergence.jl")
include("irls.jl")
include("poisson.jl")
include("logit.jl")
include("probit.jl")
include("clogit.jl")
include("ols.jl")
include("cre.jl")
include("iv.jl")
include("etwfe.jl")

# --- API ---

function feglm(df::DataFrame, formula::FormulaTerm; family = :poisson, vcov = Vcov.simple(), kwargs...)
    if family == :poisson
        return fepois(df, formula; vcov = vcov, kwargs...)
    elseif family == :logit
        return felogit(df, formula; vcov = vcov, kwargs...)
    elseif family == :probit
        return feprobit(df, formula; vcov = vcov, kwargs...)
    elseif family == :gaussian || family == :ols
        return feols(df, formula; vcov = vcov, kwargs...)
    else
        throw(ArgumentError("Unknown family: $family"))
    end
end

end
