using LinearAlgebra
using Printf
using JLD2
using IterativeSolvers
using LinearMaps
using DataStructures


# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../MA_best/FLOP_count.jl")

function correction_equations_minres(A, U, lambdas, R; tol=1e-1, maxiter=100)
    global NFLOPs
    n, k = size(U)
    S = zeros(eltype(A), n, k)
    total_iter = 0

    for j in 1:k
        λ, r = lambdas[j], R[:, j]

        M_apply = function(x)
            x_perp = x - (U * (U' * x)); 
            count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

            tmp = (A * x_perp) - λ * x_perp; 
            count_matmul_flops(n,1,n); count_vec_scaling_flops(n); count_vec_add_flops(n)

            res = tmp - (U * (U' * tmp)); 
            count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)
            return res
        end

        M_op = LinearMap{eltype(A)}(M_apply, n, n; ishermitian=true)

        rhs = r - (U * (U' * r)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)
        rhs = -rhs; count_vec_scaling_flops(n)

        s_j, msg = minres(M_op, rhs; reltol=tol, maxiter=maxiter, log=true)
        m = match(r"(\d+)\s+iterations", string(msg))
        if m !== nothing
            niter = parse(Int, m.captures[1])
            total_iter += niter
            # println("Number of iterations: ", niter)
        else
            println("No iteration number found in message: ", msg)
        end

        s_j = s_j - (U * (U' * s_j)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

        S[:, j] = s_j
    end
    println("Total MINRES iterations: ", total_iter)
    NFLOPs += total_iter * (2*n^2 + 4*n*k)  
    return S
end


function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    n, ν = size(t_candidates)
    n_b = 0

    # Preallocate maximum possible output (we trim at the end)
    T_hat = Matrix{eltype(t_candidates)}(undef, n, ν)

    # Build one combined orthogonalization basis
    if size(V_lock,2) > 0
        W = hcat(V, V_lock)
    else
        W = V
    end

    have_W = size(W,2) > 0   # for a cheap check

    for i in 1:ν
        t_i = @view t_candidates[:, i]

        # Early norm (avoids unnecessary work)
        old_norm = norm(t_i)
        count_norm_flops(length(t_i))
        if old_norm < droptol
            continue
        end

        # Copy so we don't mutate original
        t = copy(t_i)

        # === Orthogonalize t against W (block Gram-Schmidt)

        # We allow up to maxorth reorthogonalizations
        for _ in 1:maxorth
            if have_W
                # coeffs = W' * t
                count_matmul_flops(size(W,2), 1, n)
                coeffs = W' * t
                # t .-= W * coeffs
                count_matmul_flops(n, 1, size(W,2))
                t .-= W * coeffs
                count_vec_add_flops(n)
            end

            new_norm = norm(t)
            count_norm_flops(length(t))
            if new_norm > η * old_norm
                old_norm = new_norm
                break
            end
            old_norm = new_norm
        end

        # Final norm check
        final_norm = norm(t)
        count_norm_flops(length(t))
        if final_norm < droptol
            continue
        end

        # Normalize and accept
        n_b += 1
        # division by scalar
        count_vec_scaling_flops(n)
        T_hat[:, n_b] = t / final_norm
    end

    return T_hat[:,1:n_b], n_b
end


function occupied_orbitals(molecule::String)
    if molecule == "H2"
        return 1
    elseif molecule == "formaldehyde"
        return 10
    elseif molecule == "uracil"
        return 21
    else
        error("Unknown molecule: $molecule")
    end
end

function load_matrix(filename::String)
    N = 29791  

    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = Hermitian(A)
    return -A
end

function read_eigenresults(molecule::String)
    output_file = "../MA_best/Simulate_CWNO/CWNO_final_tilde_results.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function degeneracy_detector(eigenvalues::AbstractVector{T}; tol = 1e-5) where T<:Number
    perm = eachindex(eigenvalues)
    vals = eigenvalues
    deg_groups = Vector{Vector{Int}}()
    used = falses(length(vals))

    for i in eachindex(vals)
        if used[i]
            continue
        end

        group = Int[]
        push!(group, perm[i])   # store original index
        used[i] = true

        for j in (i+1):length(vals)
            if !used[j] && abs(vals[i] - vals[j])/max(abs(vals[i]), abs(vals[j])) < tol
                push!(group, perm[j])
                used[j] = true
            end
        end

        # Skip non-degenerate groups (groups of size 1)
        if !(length(group) == 1)
            push!(deg_groups, group)
        end
    end

    return deg_groups
end


function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer,
    all_idxs::Vector{Int})::Tuple{Vector{Float64}, Matrix{T}} where T<:AbstractFloat

    n = size(A, 1)
    n_b = size(V, 2)
    l_buffer = max(1, round(Int, l * 1.3))
    lc = max(1, round(Int, 1.005 * l))
    nu_0 = max(l_buffer, n_b)
    nevf = 0
    Nlow = size(V, 2)

    println("Starting Davidson with n_aux = $n_aux, l_buffer = $l_buffer, lc = $lc, thresh = $thresh, max_iter = $max_iter")

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, n, 0)
    V_lock = Matrix{T}(undef, n, 0)

    iter = 0

    # === NEW: Dictionary-based residual tracking ===
    # Maps ritz_id -> (eigenvalue, residual_norm, first_iteration_seen)
    ritz_history = Dict{Int, NamedTuple{(:lambda, :residual, :iter), Tuple{Float64, Float64, Int}}}()
    next_ritz_id = 1
    
    # Map from current sorted position -> ritz_id
    position_to_id = Dict{Int, Int}()

    # Ensure V is full rank / orthonormal initially
    if size(V,2) == 0
        error("Initial subspace V must have at least one column.")
    end

    while nevf < l_buffer
        iter += 1
        n_c = 0
        if iter > max_iter
            println("Max iterations ($max_iter) reached without full expected convergence. Returning what we have.")
            return (Eigenvalues, Ritz_vecs)
        end

        # Orthogonalize against locked vectors
        if size(V_lock, 2) > 0
            V_lock = Matrix(qr(V_lock).Q)
            count_qr_flops(n, size(V_lock,2))
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    count_dot_product_flops(n)
                    count_vec_scaling_flops(n)
                    count_vec_add_flops(n)
                    V[:, j] .-= v_lock * (v_lock' * V[:, j])
                end
            end
        end

        # Orthonormalize the basis
        V = Matrix(qr(V).Q)
        count_qr_flops(n, size(V,2))

        # Rayleigh-Ritz
        AV = A * V
        count_matmul_flops(n, size(V,2), n)
        H = Hermitian(V' * AV)
        count_matmul_flops(size(V,2), size(AV,2), n)

        nu = min(n_aux÷4, size(H,1), nu_0 - nevf)
        count_diag_flops(nu)
        Σ, U = eigen(H, 1:nu)
        X = V * U
        count_matmul_flops(n, nu, size(V,2))

        AX = A * X
        count_matmul_flops(n, nu, n)
        for col in 1:nu
            count_vec_scaling_flops(n)
        end
        R = X .* Σ' .- AX
        for col in 1:nu
            count_vec_add_flops(n)
        end

        norms = vec(norm.(eachcol(R)))
        for _ in 1:nu
            count_norm_flops(n)
        end

        # Sort Ritz values ascending
        sorted_indices = sortperm(Σ)
        Σ_sorted = Σ[sorted_indices]
        X_sorted = X[:, sorted_indices]
        R_sorted = R[:, sorted_indices]
        norms_sorted = norms[sorted_indices]

        # === NEW: Update position_to_id mapping ===
        new_position_to_id = Dict{Int, Int}()
        
        for (new_pos, old_pos) in enumerate(sorted_indices)
            # Check if this old position had an ID assigned
            if haskey(position_to_id, old_pos)
                # Reuse the existing ID
                id = position_to_id[old_pos]
                new_position_to_id[new_pos] = id
            else
                # Assign a new ID
                new_position_to_id[new_pos] = next_ritz_id
                next_ritz_id += 1
            end
        end
        
        position_to_id = new_position_to_id

        # === NEW: Update ritz_history with current iteration data ===
        for pos in 1:length(Σ_sorted)
            id = position_to_id[pos]
            ritz_history[id] = (lambda=Σ_sorted[pos], residual=norms_sorted[pos], iter=iter)
        end

        current_cutoff = min(lc - nevf, length(Σ_sorted))
        deg_groups = degeneracy_detector(Σ_sorted; tol = 1e-3)
        locked_sorted_positions = Int[]

        # Helper function to remove locked vectors from tracking
        function remove_locked_id(pos::Int)
            if haskey(position_to_id, pos)
                id = position_to_id[pos]
                delete!(ritz_history, id)
                delete!(position_to_id, pos)
            end
        end

        # --- Lock degenerate clusters ---
        for group in deg_groups
            inside = filter(i -> i <= current_cutoff, group)
            outside = filter(i -> i > current_cutoff, group)
            res_ok_inside = all(norms_sorted[i] < thresh for i in inside)
            res_ok = all(norms_sorted[i] < thresh for i in group)

            if isempty(inside)
                continue
            elseif isempty(outside)
                if res_ok
                    for gi in group
                        λ = Σ_sorted[gi]; xvec = X_sorted[:, gi]
                        push!(Eigenvalues, float(λ))
                        Ritz_vecs = hcat(Ritz_vecs, xvec)
                        V_lock = hcat(V_lock, xvec)
                        push!(locked_sorted_positions, gi)
                        nevf += 1; n_c += 1
                        remove_locked_id(gi)
                    end
                end
                continue
            else
                if res_ok_inside
                    for gi in inside
                        λ = Σ_sorted[gi]; xvec = X_sorted[:, gi]
                        push!(Eigenvalues, float(λ))
                        Ritz_vecs = hcat(Ritz_vecs, xvec)
                        V_lock = hcat(V_lock, xvec)
                        push!(locked_sorted_positions, gi)
                        nevf += 1; n_c += 1
                        remove_locked_id(gi)
                    end
                end
            end
        end

        # --- Lock isolated eigenvalues ---
        for i in 1:current_cutoff
            if i in locked_sorted_positions
                continue
            end
            if norms_sorted[i] < thresh
                λ = Σ_sorted[i]; xvec = X_sorted[:, i]
                push!(Eigenvalues, float(λ))
                Ritz_vecs = hcat(Ritz_vecs, xvec)
                V_lock = hcat(V_lock, xvec)
                push!(locked_sorted_positions, i)
                nevf += 1; n_c += 1
                remove_locked_id(i)
            end
        end

        if nevf >= lc
            println("Converged all required eigenvalues (cluster-aware). Iter = $iter")
            return (Eigenvalues, Ritz_vecs)
        end

        # --- Prepare candidates for corrections ---
        all_sorted_positions = collect(1:length(Σ_sorted))
        non_conv_positions = setdiff(all_sorted_positions, locked_sorted_positions)
        non_conv_sorted_positions = sort(non_conv_positions, by=i->Σ_sorted[i])
        nbuffer = max(1, l_buffer - nevf)
        keep_positions = non_conv_sorted_positions[1:min(length(non_conv_sorted_positions), nbuffer)]
        X_nc = X_sorted[:, keep_positions]
        Σ_nc = Σ_sorted[keep_positions]
        R_nc = R_sorted[:, keep_positions]

        # === NEW: Stagnation detection using history ===
        function is_stagnating_improved(pos::Int; rel_tol=0.1, min_iters=3)
            if !haskey(position_to_id, pos)
                return false
            end
            id = position_to_id[pos]
            if !haskey(ritz_history, id)
                return false
            end
            
            current_data = ritz_history[id]
            current_residual = current_data.residual
            
            # Check if this vector has been around long enough
            iters_tracked = iter - current_data.iter + 1
            if iters_tracked < min_iters
                return false
            end
            
            # Get residual from 2-3 iterations ago (if available)
            # We approximate by checking if residual hasn't improved much
            # A more sophisticated approach would store full history
            
            # For now, we use a simple heuristic:
            # If residual is large and hasn't converged in several iterations, it's stagnating
            if current_residual > 1e-2 && iters_tracked >= min_iters
                return true
            end
            
            return false
        end

        # --- Compute correction vectors ---
        ϵ = 1e-8
        t = zeros(T, n, length(keep_positions))

        if iter < 100
            for (i_local, pos) in enumerate(keep_positions)
                denom = clamp.(Σ_nc[i_local] .- D, ϵ, Inf)
                t[:, i_local] = R_nc[:, i_local] ./ denom
                count_vec_scaling_flops(n)
            end
        else
            # Hybrid Davidson-JD
            dav_indices = Int[]
            jd_indices = Int[]
            
            for (i_local, pos) in enumerate(keep_positions)
                rnorm = norms_sorted[pos]
                
                # === NEW: Use improved stagnation detection ===
                if rnorm >= 1e-2 || is_stagnating_improved(pos)
                    push!(jd_indices, i_local)
                else
                    push!(dav_indices, i_local)
                end
            end

            # Davidson corrections
            t_dav = zeros(T, n, length(dav_indices))
            for (j, i_local) in enumerate(dav_indices)
                denom = clamp.(Σ_nc[i_local] .- D, ϵ, Inf)
                t_dav[:, j] = R_nc[:, i_local] ./ denom
                count_vec_scaling_flops(n)
            end

            # JD corrections
            if !isempty(jd_indices)
                X_jd = X_nc[:, jd_indices]
                Σ_jd = Σ_nc[jd_indices]
                R_jd = R_nc[:, jd_indices]
                t_jd = correction_equations_minres(A, X_jd, Σ_jd, R_jd; tol=1e-1, maxiter=25)
            else
                t_jd = zeros(T, n, 0)
            end

            t = hcat(t_dav, t_jd)
        end

        # Print iteration info
        i_max = argmin(Σ_sorted)
        norm_largest_Ritz = norms_sorted[i_max]
        println("Iter $iter: V_size = $n_b, Converged = $nevf, ‖r‖ (largest λ) = $norm_largest_Ritz")

        # --- Orthonormalize & update V ---
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-12)
        if size(V, 2) + n_b_hat > n_aux && n_c > 0
            extra_idx = all_idxs[Nlow+1+(nevf-n_c) : Nlow+nevf]
            V = hcat(X_nc, T_hat, A[:, extra_idx])

        elseif size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            V = hcat(X_nc, T_hat)

        elseif n_c > 0
            extra_idx = all_idxs[Nlow+1+(nevf-n_c) : Nlow+nevf]
            V = hcat(V, T_hat, A[:, extra_idx])

        else
            V = hcat(V, T_hat)
        end

        n_b = size(V, 2)
    end

    return (Eigenvalues, Ritz_vecs)
end

function main(molecule::String, l::Integer, Naux::Integer, max_iter::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../MA_best/Simulate_CWNO/CWNO_final_tilde.dat"
    Nlow = Naux ÷ 4
    A = load_matrix(filename)
    D = diag(A)
    all_idxs = sortperm(abs.(D), rev = true)
    V = A[:, all_idxs[1:Nlow]] # only use the first Nlow columns of A as initial guess

    @time Σ, U = davidson(A, V, Naux, l, 1e-4, max_iter, all_idxs)

    idx = sortperm(Σ)
    Σ = abs.(Σ[idx])
       
    println("Number of FLOPs: $NFLOPs")

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact = read_eigenresults(molecule)
    Σexact = abs.(Σexact)
    idx_exact = sortperm(Σexact, rev=true)
    Σexact = Σexact[idx_exact]


    # Display difference
    r = min(length(Σ), l)
    println("\nCompute the difference between computed and exact eigenvalues:")
    display("text/plain", (Σ[1:r] - Σexact[1:r])')
    # difference = (Σ[1:r] .- Σexact[1:r])
    # for i in 1:r
    #     println(@sprintf("%3d: %.10f (computed) - %.10f (exact) = % .4e", i, Σ[i], Σexact[i], difference[i]))
    # end
    println("$r Eigenvalues converges, out of $l requested.")
end

molecule_dict = Dict(
    "formaldehyde" => [10, 50, 100, 200]
)

for mol in keys(molecule_dict)
    println("\n=== Running tests for molecule: $mol ===")
    for l in molecule_dict[mol]
        nev = l*occupied_orbitals(mol)
        Naux = 4 * nev
        println("\n--- Running Davidson for $nev eigenvalues (Naux = $Naux) ---")
        main(mol, nev, Naux, 100)
    end
end


