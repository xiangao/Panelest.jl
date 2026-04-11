"""
Highly optimized demeaning algorithm ported from fixest (demeaning.cpp).
Implements iterative projection with Irons-Tuck acceleration.
"""

struct DemeanSolver
    fes_refs::Vector{Vector{Int32}}
    fe_levels::Vector{Int}
    fe_order::Vector{Int}
    n_obs::Int
    # Cache for weights
    cached_weights::Vector{Float64}
    cached_level_weights::Vector{Vector{Float64}}
    
    function DemeanSolver(fes_refs::Vector{Vector{Int32}}, n_obs::Int)
        fe_levels = [Int(maximum(refs)) for refs in fes_refs]
        fe_order = sortperm(fe_levels, rev=true)
        new(fes_refs, fe_levels, fe_order, n_obs, Float64[], [Float64[] for _ in fes_refs])
    end
end

function DemeanSolver(fes_refs::Vector{<:AbstractVector{<:Integer}}, n_obs::Int)
    refs32 = [convert(Vector{Int32}, r) for r in fes_refs]
    return DemeanSolver(refs32, n_obs)
end

function get_level_weights!(solver::DemeanSolver, weights::AbstractVector{Float64}, q_idx::Int)
    n_levels = solver.fe_levels[q_idx]
    
    if solver.cached_weights === weights && !isempty(solver.cached_level_weights[q_idx])
        return solver.cached_level_weights[q_idx]
    end
    
    if length(solver.cached_level_weights[q_idx]) != n_levels
        solver.cached_level_weights[q_idx] = zeros(Float64, n_levels)
    else
        fill!(solver.cached_level_weights[q_idx], 0.0)
    end
    
    lw = solver.cached_level_weights[q_idx]
    refs = solver.fes_refs[q_idx]
    @inbounds for i in 1:solver.n_obs
        lw[refs[i]] += weights[i]
    end
    
    return lw
end

function demean_fixest!(
    X::AbstractVecOrMat{Float64},
    weights::AbstractVector{Float64},
    solver::DemeanSolver;
    tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    accelerate::Bool = true
)
    n = solver.n_obs
    n_vars = size(X, 2)
    
    # Each thread needs its own buffers if we parallelize at column level
    # But demean_fixest! might be called from solve_residuals_fixest! which is already threaded.
    # So we use thread-local buffers or just allocate here.
    max_levels = maximum(solver.fe_levels)
    level_sum = zeros(Float64, max_levels)
    v_prev = zeros(Float64, n)
    v_prevprev = zeros(Float64, n)
    
    all_level_weights = [get_level_weights!(solver, weights, q) for q in 1:length(solver.fes_refs)]
    
    for j in 1:n_vars
        v = v_view(X, j)
        demean_column_optimized!(v, weights, solver, level_sum, v_prev, v_prevprev, all_level_weights; 
                                 tol=tol, maxiter=maxiter, accelerate=accelerate)
    end
    
    return X
end

@inline v_view(X::AbstractMatrix, j) = view(X, :, j)
@inline v_view(X::AbstractVector, j) = X

function demean_column_optimized!(
    v::AbstractVector{Float64},
    weights::AbstractVector{Float64},
    solver::DemeanSolver,
    level_sum::Vector{Float64},
    v_prev::Vector{Float64},
    v_prevprev::Vector{Float64},
    all_level_weights::Vector{Vector{Float64}};
    tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    accelerate::Bool = true
)
    n = solver.n_obs
    copyto!(v_prev, v)
    copyto!(v_prevprev, v)
    
    converged = false
    
    for iter in 1:maxiter
        copyto!(v_prevprev, v_prev)
        copyto!(v_prev, v)
        
        for q_idx in solver.fe_order
            refs = solver.fes_refs[q_idx]
            n_levels = solver.fe_levels[q_idx]
            lw = all_level_weights[q_idx]
            
            @inbounds for i in 1:n_levels
                level_sum[i] = 0.0
            end
            
            @inbounds for i in 1:n
                level_sum[refs[i]] += weights[i] * v[i]
            end
            
            @inbounds for i in 1:n
                l = refs[i]
                if lw[l] > 1e-15
                    v[i] -= level_sum[l] / lw[l]
                end
            end
        end
        
        max_diff = 0.0
        @inbounds for i in 1:n
            diff = abs(v[i] - v_prev[i])
            if diff > max_diff
                max_diff = diff
            end
        end
        
        if max_diff < tol
            converged = true
            break
        end
        
        if accelerate && iter > 2 && iter % 3 == 0
            dot_delta_delta2 = 0.0
            dot_delta2_delta2 = 0.0
            @inbounds for i in 1:n
                delta = v[i] - v_prev[i]
                delta2 = v[i] - 2*v_prev[i] + v_prevprev[i]
                dot_delta_delta2 += delta * delta2
                dot_delta2_delta2 += delta2 * delta2
            end
            if dot_delta2_delta2 > 1e-15
                coef = dot_delta_delta2 / dot_delta2_delta2
                @inbounds for i in 1:n
                    v[i] -= coef * (v[i] - v_prev[i])
                end
            end
        end
    end
    
    return converged
end

struct TwoFEGauSolver
    n_i::Int
    n_j::Int
    n_cells::Int
    mat_row::Vector{Int32}
    mat_col::Vector{Int32}
    mat_value::Vector{Float64} # Cell weights
    ca::Vector{Float64} # Inverse level weights for FE 1
    cb::Vector{Float64} # Inverse level weights for FE 2
    obs_to_cell::Vector{Int}
    
    function TwoFEGauSolver(weights::Vector{Float64}, fes_refs::Vector{Vector{Int32}}, fe_levels::Vector{Int})
        n_i, n_j = fe_levels[1], fe_levels[2]
        n_obs = length(weights)
        cell_map = Dict{Int64, Int}()
        mat_row, mat_col, mat_value = Int32[], Int32[], Float64[]
        obs_to_cell = zeros(Int, n_obs)
        lw_i, lw_j = zeros(Float64, n_i), zeros(Float64, n_j)
        
        refs_i, refs_j = fes_refs[1], fes_refs[2]
        @inbounds for k in 1:n_obs
            i, j, w = refs_i[k], refs_j[k], weights[k]
            lw_i[i] += w
            lw_j[j] += w
            key = Int64(i) | (Int64(j) << 32)
            idx = get(cell_map, key, 0)
            if idx == 0
                push!(mat_row, i); push!(mat_col, j); push!(mat_value, w)
                idx = length(mat_row); cell_map[key] = idx
            else
                mat_value[idx] += w
            end
            obs_to_cell[k] = idx
        end
        ca = [w > 1e-15 ? 1.0/w : 0.0 for w in lw_i]
        cb = [w > 1e-15 ? 1.0/w : 0.0 for w in lw_j]
        new(n_i, n_j, length(mat_row), mat_row, mat_col, mat_value, ca, cb, obs_to_cell)
    end
end

function demean_2fe_full!(v::AbstractVector{Float64}, weights::AbstractVector{Float64}, 
                         solver::TwoFEGauSolver, refs::Vector{Vector{Int32}};
                         tol::Float64 = 1e-8, maxiter::Int = 1000)
    n_i, n_j, n_cells, n_obs = solver.n_i, solver.n_j, solver.n_cells, length(v)
    mat_row, mat_col, mat_value = solver.mat_row, solver.mat_col, solver.mat_value
    ca, cb, obs_to_cell = solver.ca, solver.cb, solver.obs_to_cell
    
    cell_v = zeros(Float64, n_cells)
    @inbounds for k in 1:n_obs; cell_v[obs_to_cell[k]] += weights[k] * v[k]; end
    
    const_a, const_b = zeros(Float64, n_i), zeros(Float64, n_j)
    @inbounds for c in 1:n_cells
        cv = cell_v[c]
        const_a[mat_row[c]] += cv
        const_b[mat_col[c]] += cv
    end
    @inbounds for i in 1:n_i; const_a[i] *= ca[i]; end
    @inbounds for j in 1:n_j; const_b[j] *= cb[j]; end
    
    a_tilde = copy(const_a)
    @inbounds for c in 1:n_cells
        a_tilde[mat_row[c]] -= (mat_value[c] * ca[mat_row[c]]) * const_b[mat_col[c]]
    end
    
    X, GX, GGX, delta_GX, delta2_X = zeros(n_i), zeros(n_i), zeros(n_i), zeros(n_i), zeros(n_i)
    tmp_beta = zeros(Float64, n_j)
    
    function apply_Ab_Ba!(dest, src)
        fill!(tmp_beta, 0.0)
        @inbounds for c in 1:n_cells
            tmp_beta[mat_col[c]] += (mat_value[c] * cb[mat_col[c]]) * src[mat_row[c]]
        end
        fill!(dest, 0.0)
        @inbounds for c in 1:n_cells
            dest[mat_row[c]] += (mat_value[c] * ca[mat_row[c]]) * tmp_beta[mat_col[c]]
        end
        @inbounds for i in 1:n_i; dest[i] += a_tilde[i]; end
    end
    
    apply_Ab_Ba!(GX, X)
    converged = false
    for iter in 1:maxiter
        max_diff = 0.0
        @inbounds for i in 1:n_i
            diff = abs(X[i] - GX[i]) / (0.1 + abs(GX[i]))
            if diff > max_diff; max_diff = diff; end
        end
        if max_diff < tol; converged = true; break; end
        apply_Ab_Ba!(GGX, GX)
        @inbounds for i in 1:n_i
            delta_GX[i] = GGX[i] - GX[i]
            delta2_X[i] = delta_GX[i] - GX[i] + X[i]
        end
        vprod, ssq = 0.0, 0.0
        @inbounds for i in 1:n_i
            vprod += delta_GX[i] * delta2_X[i]
            ssq += delta2_X[i] * delta2_X[i]
        end
        if ssq > 1e-15
            coef = vprod / ssq
            @inbounds for i in 1:n_i; X[i] = GGX[i] - coef * delta_GX[i]; end
        else
            converged = true; break
        end
        apply_Ab_Ba!(GX, X)
    end
    
    beta = copy(const_b)
    @inbounds for c in 1:n_cells
        beta[mat_col[c]] -= (mat_value[c] * cb[mat_col[c]]) * GX[mat_row[c]]
    end
    
    ref1, ref2 = refs[1], refs[2]
    @inbounds for k in 1:n_obs
        v[k] -= GX[ref1[k]] + beta[ref2[k]]
    end
end

function solve_residuals_fixest!(cols, fes, weights; tol = 1e-8, maxiter = 1000, solver = nothing)
    n = length(weights)
    fes_refs = [convert(Vector{Int32}, fe.refs) for fe in fes]
    
    if length(fes) == 2
        fe2_solver = TwoFEGauSolver(weights, fes_refs, [Int(maximum(r)) for r in fes_refs])
        Threads.@threads for col in cols
            demean_2fe_full!(col, weights, fe2_solver, fes_refs; tol=tol, maxiter=maxiter)
        end
        return
    end

    if solver === nothing
        solver = DemeanSolver(fes_refs, n)
    end
    
    Threads.@threads for col in cols
        demean_fixest!(col, weights, solver; tol = tol, maxiter = maxiter)
    end
end
