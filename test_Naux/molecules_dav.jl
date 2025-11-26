using LinearAlgebra
using Printf
using JLD2
using IterativeSolvers
using LinearMaps
using DataStructures


# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../../MA_best/FLOP_count.jl")

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
        return 6
    elseif molecule == "uracil"
        return 21
    else
        error("Unknown molecule: $molecule")
    end
end

function load_matrix(filename::String, molecule::String)
    if molecule == "H2"
        N = 11994
    elseif molecule == "formaldehyde"
        N = 27643
    elseif molecule == "uracil"
        N = 32416
    else
        error("Unknown molecule: $molecule")
    end
    # println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

function read_eigenresults(molecule::String)
    output_file = "../../MA_best/Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
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

# --- small helper (your provided stagnation test) ---
function is_stagnating(hist::Vector{Float64}; tol=0.1, window=2)
    length(hist) < window && return false
    r_old, r_new = hist[end-window+1], hist[end]
    # avoid division by zero
    if r_old == 0.0
        return r_new == 0.0
    end
    return abs(r_old - r_new) / r_old < tol
end
        

# --- Replace your existing davidson(...) with this updated version ---
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

    # === NEW: richer ritz history tracking ===
    # ritz_history: Dict{Int, NamedTuple{(:lambda_hist, :res_hist, :first_iter), Tuple{Vector{Float64}, Vector{Float64}, Int}}}
    ritz_history = Dict{Int, NamedTuple{(:lambda_hist, :res_hist, :first_iter), Tuple{Vector{Float64}, Vector{Float64}, Int}}}()
    next_ritz_id = 1

    # parameters for matching and history
    match_tol = 1e-6          # tolerance for matching eigenvalues across iterations (can be tuned)
    history_window = 5        # keep at most this many residual history entries per tracked ritz

    # helper to find best match for a lambda among existing ritz_history keys
    function find_best_match(λ::Float64)
        best_id = nothing
        best_dist = Inf

        for (id, data) in ritz_history
            # skip entries with empty history
            if isempty(data.lambda_hist)
                continue
            end

            last_lambda = data.lambda_hist[end]
            dist = abs(last_lambda - λ)

            if dist < best_dist
                best_dist = dist
                best_id = id
            end
        end

        if best_id === nothing
            return nothing
        end

        last_lambda = ritz_history[best_id].lambda_hist[end]

        # Corrected denom expression
        denom = max(abs(λ), abs(last_lambda), 1.0)

        # relative tolerance check
        if best_dist / denom < max(match_tol, 1e-8)
            return best_id
        else
            return nothing
        end
    end


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

        nu = min(n_aux÷6, size(H,1), nu_0 - nevf)
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

        # === NEW: robust matching of current ritzs to tracked IDs by lambda closeness ===
        # We'll build mapping current_pos -> ritz_id
        current_pos_to_id = Dict{Int, Int}()

        # Temporary set to avoid assigning one tracked id to multiple current positions
        used_ids = Set{Int}()

        for pos in 1:length(Σ_sorted)
            λ = Σ_sorted[pos]
            matched_id = find_best_match(λ)
            if matched_id !== nothing && !(matched_id in used_ids)
                # assign existing id
                current_pos_to_id[pos] = matched_id
                push!(used_ids, matched_id)
            else
                # create new id
                new_id = next_ritz_id
                next_ritz_id += 1
                current_pos_to_id[pos] = new_id
                # initialize history entry
                ritz_history[new_id] = (lambda_hist = Float64[], res_hist = Float64[], first_iter = iter)
                push!(used_ids, new_id)
            end
        end

        # Append current data to histories (and cap sizes)
        for pos in 1:length(Σ_sorted)
            id = current_pos_to_id[pos]
            data = ritz_history[id]
            # append
            push!(data.lambda_hist, Σ_sorted[pos])
            push!(data.res_hist, norms_sorted[pos])
            # cap histories
            if length(data.lambda_hist) > history_window
                data = (lambda_hist = data.lambda_hist[end-history_window+1:end],
                        res_hist = data.res_hist[end-history_window+1:end],
                        first_iter = data.first_iter)
            end
            # store back
            ritz_history[id] = data
        end

        current_cutoff = min(lc - nevf, length(Σ_sorted))
        deg_groups = degeneracy_detector(Σ_sorted; tol = 1e-3)
        locked_sorted_positions = Int[]

        # Helper function to remove locked vectors from tracking
        function remove_locked_id(pos::Int)
            if haskey(current_pos_to_id, pos)
                id = current_pos_to_id[pos]
                # remove only the id (we may have other pos->id mappings for same id in pathological cases)
                if haskey(ritz_history, id)
                    delete!(ritz_history, id)
                end
                # remove mapping
                delete!(current_pos_to_id, pos)
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

        # === NEW: Stagnation detection using history (from ritz_history) ===
        function is_stagnating_improved_for_pos(pos::Int; rel_tol=0.1, min_iters=2)
            # get tracked id for this pos
            if !haskey(current_pos_to_id, pos)
                return false
            end
            id = current_pos_to_id[pos]
            if !haskey(ritz_history, id)
                return false
            end
            hist = ritz_history[id].res_hist
            return is_stagnating(hist; tol=rel_tol, window=min(min_iters, length(hist)))
        end

        # --- Compute correction vectors ---
        ϵ = 1e-8
        t = zeros(T, n, length(keep_positions))

        if iter < 8
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
                # pos is the position in the sorted list; norms_sorted uses that indexing
                rnorm = norms_sorted[pos]
                # Use residual history-based test
                if rnorm >= 1e-2 || is_stagnating_improved_for_pos(pos; rel_tol=0.1, min_iters=2)
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
            extra_idx = all_idxs[(Nlow+1+(nevf - n_c)) : (Nlow+nevf)]
            if size(X_nc, 2) == 0
                println("Warning: X_nc is empty when rebuilding V, using only T_hat, size = $(size(T_hat)) and extra A columns, size = $(size(A[:, extra_idx])).")
            end
            V = hcat(X_nc, T_hat, A[:, extra_idx])

        elseif size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            V = hcat(X_nc, T_hat)

        elseif n_c > 0
            extra_idx = all_idxs[(Nlow+1+(nevf - n_c)) : (Nlow+nevf)]
            V = hcat(V, T_hat, A[:, extra_idx])

        else
            V = hcat(V, T_hat)
        end

        n_b = size(V, 2)

        if size(V, 2)==0
            println("Warning: V is empty, rebuilding from A columns.")
            extra_idx = all_idxs[(Nlow+1+(nevf - n_c)): (Nlow+nevf + Nlow)] # take some extra columns to avoid empty V
            V = A[:, extra_idx]
        end
    end

    return (Eigenvalues, Ritz_vecs)
end


function main(molecule::String, l::Integer, Naux::Integer, max_iter::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../../MA_best/" * molecule *"/gamma_VASP_RNDbasis1.dat"
    Nlow = Naux ÷ 6
    A = load_matrix(filename,molecule)
    D = diag(A)
    all_idxs = sortperm(abs.(D), rev = true)
    V = A[:, all_idxs[1:Nlow]] # only use the first Nlow columns of A as initial guess

    if molecule == "H2"
        accuracy = 1e-5
    else
        accuracy = 1e-3
    end

    @time Σ, U = davidson(A, V, Naux, l, accuracy, max_iter, all_idxs)

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

Naux = [400, 500, 600, 700] #600, 1200, 2400
ls = [50, 100, 200] #10, 50, 100, 
molecule = "H2" # 'uracil', 'H2',

for d in Naux
    println("\n=== Running tests for molecule: $molecule ===")
    for l in ls
        nev = l*occupied_orbitals(molecule)
        main(molecule, nev, d, 100)
    end
end
