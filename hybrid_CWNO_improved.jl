using LinearAlgebra
using Printf
using JLD2
using IterativeSolvers
using LinearMaps


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
        if old_norm < droptol
            continue
        end

        # Copy so we don't mutate original
        t = copy(t_i)

        # === Orthogonalize t against W (block Gram-Schmidt)
        # We allow up to maxorth reorthogonalizations
        for _ in 1:maxorth
            if have_W
                coeffs = W' * t
                t .-= W * coeffs
            end

            # reorthogonalize against already-accepted T_hat(1:n_b)
            if n_b > 0
                Th = @view T_hat[:,1:n_b]
                coeffs2 = Th' * t
                t .-= Th * coeffs2
            end

            new_norm = norm(t)
            if new_norm > η * old_norm
                old_norm = new_norm
                break
            end
            old_norm = new_norm
        end

        # Final norm check
        final_norm = norm(t)
        if final_norm < droptol
            continue
        end

        # Normalize and accept
        n_b += 1
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

function is_too_close_to_converged(λ, Eigenvalues, tol_rel)
    for λc in Eigenvalues
        if abs(λ - λc) ≤ tol_rel * max(abs(λ), abs(λc))
            return true
        end
    end
    return false
end


function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer,
    stable_thresh::Integer = 3  # kept for API compatibility but not used for locking
)::Tuple{Vector{Float64}, Matrix{T}} where T<:AbstractFloat

    n = size(A, 1)
    n_b = size(V, 2)
    l_buffer = max(1, round(Int, l * 1.3))
    lc = max(1, round(Int, 1.005 * l))  # we want to converge smallest lc eigenvalues
    nu_0 = max(l_buffer, n_b)
    nevf = 0

    println("Starting Davidson with n_aux = $n_aux, l_buffer = $l_buffer, lc = $lc, thresh = $thresh, max_iter = $max_iter")

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, n, 0)
    V_lock = Matrix{T}(undef, n, 0)

    iter = 0
    residual_history = Dict{Int, Vector{Float64}}()   # for stagnation detection

    # Helper function: stagnation check
    function is_stagnating(hist::Vector{Float64}; tol=0.1, window=2)
        length(hist) < window && return false
        r_old, r_new = hist[end-window+1], hist[end]
        return abs(r_old - r_new) / r_old < tol
    end

    # Ensure V is full rank / orthonormal initially
    if size(V,2) == 0
        error("Initial subspace V must have at least one column.")
    end

    while nevf < l_buffer
        iter += 1
        if iter > max_iter
            println("Max iterations ($max_iter) reached without full expected convergence. Returning what we have.")
            return (Eigenvalues, Ritz_vecs)
        end

        # Orthogonalize against locked vectors
        if size(V_lock, 2) > 0
            V_lock = Matrix(qr(V_lock).Q)
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    V[:, j] .-= v_lock * (v_lock' * V[:, j])
                end
            end
        end

        # Orthonormalize the basis
        V = Matrix(qr(V).Q)

        # Rayleigh–Ritz
        AV = A * V
        H = Hermitian(V' * AV)

        # Compute approximate eigenvalues
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)

        # Compute Ritz vectors
        X = V * U

        # Residuals R = A X - X Σ  (we will keep sign consistent with your earlier code)
        R = X .* Σ' .- A * X
        
        # residual norms
        norms = vec(norm.(eachcol(R)))

        # sort by Ritz value (ascending)
        sorted_indices = sortperm(Σ)
        Σ_sorted = Σ[sorted_indices]
        X_sorted = X[:, sorted_indices]
        R_sorted = R[:, sorted_indices]
        norms_sorted = norms[sorted_indices]

        # how many eigenvalues to consider for convergence this outer iteration
        current_cutoff = min(lc - nevf, length(Σ_sorted))

        # === New: degeneracy-based locking ===
        # detect degenerate clusters (groups of indices in 1:length(Σ_sorted))
        deg_groups = degeneracy_detector(Σ_sorted; tol = 1e-3)

        locked_sorted_positions = Int[]  # positions in the sorted arrays that were locked this iteration

        # Lock entire clusters only if every member has residual norm < thresh

        for group in deg_groups
            inside = filter(i -> i <= current_cutoff, group)
            outside = filter(i -> i >  current_cutoff, group)

            # cluster residuals
            res_ok = all(norms_sorted[i] < thresh for i in group)
            res_ok_inside = all(norms_sorted[i] < thresh for i in inside)

            # Case C: cluster entirely outside region we currently need → skip
            if isempty(inside)
                continue
            end

            # Case A: cluster fully inside region of interest
            if isempty(outside)
                if res_ok
                    for gi in group
                        λ = Σ_sorted[gi]
                        xvec = X_sorted[:, gi]
                        push!(Eigenvalues, float(λ))
                        Ritz_vecs = hcat(Ritz_vecs, xvec)
                        V_lock = hcat(V_lock, xvec)
                        push!(locked_sorted_positions, gi)
                        nevf += 1
                    end
                end
                continue
            end

            # Case B: cluster partially inside cutoff → SPLIT cluster

            # Locking the last eigenvalue(s)
            last_eval_requested = (current_cutoff == lc - nevf)

            if last_eval_requested && res_ok
                # Allowed to lock full cluster
                for gi in group
                    λ = Σ_sorted[gi]
                    xvec = X_sorted[:, gi]
                    push!(Eigenvalues, float(λ))
                    Ritz_vecs = hcat(Ritz_vecs, xvec)
                    V_lock = hcat(V_lock, xvec)
                    push!(locked_sorted_positions, gi)
                    nevf += 1
                end
            else
                # Lock only the inside portion
                if res_ok_inside
                    for gi in inside
                        λ = Σ_sorted[gi]
                        xvec = X_sorted[:, gi]
                        push!(Eigenvalues, float(λ))
                        Ritz_vecs = hcat(Ritz_vecs, xvec)
                        V_lock = hcat(V_lock, xvec)
                        push!(locked_sorted_positions, gi)
                        nevf += 1
                    end
                end
            end
        end

        # After cluster locking, handle isolated (non-degenerate) eigenvalues individually
        for i in 1:current_cutoff
            if i in locked_sorted_positions
                continue
            end
            if norms_sorted[i] < thresh
                λ = Σ_sorted[i]
                xvec = X_sorted[:, i]
                push!(Eigenvalues, float(λ))
                Ritz_vecs = hcat(Ritz_vecs, xvec)
                V_lock = hcat(V_lock, xvec)
                push!(locked_sorted_positions, i)
                nevf += 1
            end
        end

        # If we've converged required low-lying eigenvalues, return
        if nevf >= lc
            println("Converged all required eigenvalues (cluster-aware). Iter = $iter")
            return (Eigenvalues, Ritz_vecs)
        end

        # Prepare for next iteration: pick non-converged sorted positions to expand
        all_sorted_positions = collect(1:length(Σ_sorted))
        non_conv_positions = setdiff(all_sorted_positions, locked_sorted_positions)

        # Sort non-converged positions by Ritz value (ascending)
        non_conv_sorted_positions = sort(non_conv_positions, by = i -> Σ_sorted[i])

        # select most promising candidates (up to buffer size)
        nbuffer = max(1, l_buffer - nevf)
        keep_positions = non_conv_sorted_positions[1:min(length(non_conv_sorted_positions), nbuffer)]

        # prepare block matrices for the kept candidates
        X_nc = X_sorted[:, keep_positions]
        Σ_nc = Σ_sorted[keep_positions]
        R_nc = R_sorted[:, keep_positions]

        # update residual histories for stagnation detection
        for (kpos, sp) in enumerate(keep_positions)
            push!(get!(residual_history, sp, Float64[]), norms_sorted[sp])
            if length(residual_history[sp]) > 3
                popfirst!(residual_history[sp])
            end
        end

        # === Compute correction vectors ===
        ϵ = 1e-8
        if iter < 15
            # pure Davidson early iterations
            t = zeros(T, n, size(X_nc, 2))
            for (i, _) in enumerate(keep_positions)
                denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
                t[:, i] = R_nc[:, i] ./ denom
                count_vec_add_flops(length(D))
                count_vec_scaling_flops(length(D))
            end
        else
            println("Hybrid Davidson-JD corrections at iteration $iter")
            dav_indices = Int[]
            jd_indices = Int[]
            for (i_local, sp) in enumerate(keep_positions)
                # sp is sorted position in Σ_sorted
                # decide Davidson vs JD
                if i_local > current_cutoff
                    push!(dav_indices, i_local)
                else
                    r = norms_sorted[sp]
                    hist = get(residual_history, sp, Float64[])
                    if r >= 1e-2 || is_stagnating(hist; tol=0.1, window=2)
                        push!(jd_indices, i_local)
                    else
                        push!(dav_indices, i_local)
                    end
                end
            end

            # Davidson corrections
            t_dav = zeros(T, n, length(dav_indices))
            for (j, i_local) in enumerate(dav_indices)
                denom = clamp.(Σ_nc[i_local] .- D, ϵ, Inf)
                t_dav[:, j] = R_nc[:, i_local] ./ denom
            end

            # JD corrections (cheap placeholder)
            if !isempty(jd_indices)
                # pick corresponding columns
                X_jd = X_nc[:, jd_indices]
                Σ_jd = Σ_nc[jd_indices]
                R_jd = R_nc[:, jd_indices]
                t_jd = correction_equations_minres(A, X_jd, Σ_jd, R_jd; tol=1e-1, maxiter=25)
            else
                t_jd = zeros(T, n, 0)
            end

            # Merge
            t = hcat(t_dav, t_jd)
        end

        # Orthonormalize and select correction vectors
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-12)
        if size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            # Update search space V if size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            max_new_vectors = max(0, n_aux - size(X_nc, 2))
            use = min(n_b_hat, max_new_vectors)

            if use == 0
                # fallback: restart with the best X_nc
                V = copy(X_nc)
                n_b = size(V, 2)
            else
                T_hat = T_hat[:, 1:use]
                V = hcat(X_nc, T_hat)
                n_b = size(V, 2)
            end
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end

        # Print iteration info
        i_max = argmin(Σ_sorted)
        norm_largest_Ritz = norms_sorted[i_max]
        println("Iter $iter: V_size = $n_b, Converged = $nevf, ‖r‖ (largest λ) = $norm_largest_Ritz")
    end

    return (Eigenvalues, Ritz_vecs)
end


function main(molecule::String, l::Integer, beta::Integer, factor::Integer, max_iter::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../MA_best/Simulate_CWNO/CWNO_final_tilde.dat"

    Nlow = max(round(Int, 0.1*l), 16)
    Naux = Nlow * beta
    A = load_matrix(filename)
    N = size(A, 1)

    # V = zeros(N, Nlow)
    # for i = 1:Nlow
    #     V[i, i] = 1.0
    # end

    D = diag(A)
    idxs = sortperm(abs.(D), rev = true)[1:Nlow]
    V = A[:, idxs]

    @time Σ, U = davidson(A, V, Naux, l, 1e-4, max_iter)

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
    # display("text/plain", (Σ[1:r] - Σexact[1:r])')
    # difference = (Σ[1:r] .- Σexact[1:r])
    # for i in 1:r
    #     println(@sprintf("%3d: %.10f (computed) - %.10f (exact) = % .4e", i, Σ[i], Σexact[i], difference[i]))
    # end
    println("$r Eigenvalues converges, out of $l requested.")
end



betas = [20] #8,16,32,64, 8,16
molecules = ["formaldehyde"] #, "uracil"
ls = [10, 50, 100, 200] #10, 50, 100, 200
for molecule in molecules
    println("Processing molecule: $molecule")
    for beta in betas
        println("Running with beta = $beta")
        for (i, l) in enumerate(ls)
	    nev = l*occupied_orbitals(molecule)
            println("Running with l = $nev")
            main(molecule, nev, beta, i, 100)
        end
    end
    println("Finished processing molecule: $molecule")
end


