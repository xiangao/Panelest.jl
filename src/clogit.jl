# Chamberlain's Conditional Logit
# This implementation uses the recursive algorithm (Gail, Lubin, and Gastwirth, 1981)
# to handle multiple treated units per group.

function clogit(df::DataFrame, formula::FormulaTerm, id::Symbol; vcov = Vcov.simple(), kwargs...)
    # 1. Parse formula
    formula_main, formula_fe = parse_fe(formula)
    
    # Force removal of intercept since it's not identified in conditional logit
    rhs_terms = AbstractTerm[t for t in eachterm(formula_main.rhs) if !(t isa ConstantTerm)]
    push!(rhs_terms, ConstantTerm(0))
    formula_main = FormulaTerm(formula_main.lhs, tuple(rhs_terms...))

    sch = schema(formula_main, df)
    f_schema = apply_schema(formula_main, sch, StatisticalModel)
    
    y = response(f_schema, df)
    X = modelmatrix(f_schema, df)
    coefnames_X = coefnames(f_schema)[2]
    if !isa(coefnames_X, Vector)
        coefnames_X = [coefnames_X]
    end
    
    n = length(y)
    p = size(X, 2)
    
    # 2. Group by id
    df_with_idx = copy(df)
    df_with_idx.row_idx = 1:n
    gdf = groupby(df_with_idx, id)
    
    @info "Using conditional logit (Chamberlain) recursive algorithm."
    
    # Initial beta from OLS or small values
    beta = X \ (y .- mean(y))
    
    # Define variables outside the loop for scope
    grad = zeros(p)
    hess = zeros(p, p)
    
    for i in 1:50 # Newton loop
        fill!(grad, 0.0)
        fill!(hess, 0.0)
        loglik = 0.0
        
        for group in gdf
            rows = group.row_idx
            y_g = y[rows]
            X_g = X[rows, :]
            
            ng = length(y_g)
            sg = Int(sum(y_g))
            
            if sg == 0 || sg == ng
                continue
            end
            
            eta_g = X_g * beta
            # Subtract max to avoid overflow
            max_eta = maximum(eta_g)
            exp_eta = exp.(eta_g .- max_eta)
            
            # Recursive calculation of A(k, m)
            # A[k, m] stores the sum of exp(sum(eta)) for combinations of m from first k
            A = zeros(ng + 1, sg + 1)
            A[:, 1] .= 1.0
            for k in 1:ng
                for m in 1:min(k, sg)
                    A[k+1, m+1] = A[k, m+1] + exp_eta[k] * A[k, m]
                end
            end
            
            denom = A[ng+1, sg+1]
            loglik += sum(eta_g[y_g .== 1]) - sg * max_eta - log(denom)
            
            # Gradient and Hessian using the same recursion principle for efficiency
            # D[k, m] = d A(k, m) / d beta (vector of size p)
            # H[k, m] = d^2 A(k, m) / d beta^2 (matrix of size p x p)
            # For simplicity in this prototype, we use the property:
            # d log(A) / d beta = E[X | combinations]
            
            # We need E[X] and Var(X) over the combinations.
            # Let's use a simpler recursive update for the gradient and Hessian components.
            # E[X]_{k, m} = (E[X]_{k-1, m} * A(k-1, m) + (E[X]_{k-1, m-1} + X_k) * exp(eta_k) * A(k-1, m-1)) / A(k, m)
            
            E = zeros(p, ng + 1, sg + 1)
            V = zeros(p, p, ng + 1, sg + 1)
            
            for k in 1:ng
                xk = X_g[k, :]
                for m in 1:min(k, sg)
                    # Update E
                    # Term 1: from combinations NOT including k
                    # Term 2: from combinations including k
                    w1 = A[k, m+1] / A[k+1, m+1]
                    w2 = (exp_eta[k] * A[k, m]) / A[k+1, m+1]
                    
                    E[:, k+1, m+1] = w1 * E[:, k, m+1] + w2 * (E[:, k, m] + xk)
                    
                    # Update V
                    # Var = E[X^2] - E[X]^2
                    # We can track E[X X'] similarly
                    V[:, :, k+1, m+1] = w1 * (V[:, :, k, m+1] + E[:, k, m+1] * E[:, k, m+1]') + 
                                       w2 * (V[:, :, k, m] + (E[:, k, m] + xk) * (E[:, k, m] + xk)') -
                                       E[:, k+1, m+1] * E[:, k+1, m+1]'
                end
            end
            
            grad .+= sum(X_g[y_g .== 1, :], dims=1)[:] .- E[:, ng+1, sg+1]
            hess .-= V[:, :, ng+1, sg+1]
        end
        
        # Newton step
        delta = pinv(hess) * grad
        beta .-= delta
        
        if norm(delta) < 1e-8
            break
        end
    end
    
    V_final = pinv(-hess)
    
    return PanelestModel(
        beta,
        V_final,
        zeros(n), 
        zeros(n), 
        zeros(n), 
        true,
        0, 
        n,
        n - p,
        formula,
        coefnames_X,
        :clogit, X, hess
    )
end

