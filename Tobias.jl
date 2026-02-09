using LinearAlgebra
using Printf
using Random
using JLD2

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../MA_best/FLOP_count.jl")

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
    output_file = "../MA_best/Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String,
    max_iter::Integer
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    global NFLOPs

    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    iter = 0

    while true
        iter += 1

        # QR-Orthogonalisierung
        count_qr_flops(size(V,1), size(V,2))
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # Rayleigh-Matrix: H = V' * (A * V)
        temp = A * V
        count_matmul_flops(size(A,1), size(A,2), size(V,2))  # A*V
        H = V' * temp
        count_matmul_flops(size(V,2), size(V,1), size(temp,2))  # V'*temp

        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        count_diag_flops(size(H,1))  # kleine Diagonalisierung
        
        X = V * U
        count_matmul_flops(size(V,1), size(V,2), size(U,2))  # V*U
        if iter > max_iter
            println("Max iterations ($max_iter) reached without full expected convergence. Returning what we have.")
            return (Σ, X)
        end

        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        count_matmul_flops(size(A,1), size(A,2), size(X,2))  # A*X
        R .-= temp2
        count_vec_add_flops(length(R))

        # Count norm calculation
        Rnorm = norm(R, 2)
        count_norm_flops(length(R))

        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)

        if Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end

        # Preconditioning
        t = similar(R)
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D)
            t[:,i] = C .* R[:,i]
            count_vec_add_flops(length(D))       # For Σ[i] .- D
            count_vec_scaling_flops(length(D))   # For the division
            count_vec_scaling_flops(length(D))   # For the multiplication
        end

        # Update V
        if size(V,2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

function main(system::String, Nlow::Int)
    global NFLOPs
    NFLOPs = 0  # Reset FLOP counter
    
    Naux = Nlow * 4
    filename = "../MA_best/" * system *"/gamma_VASP_RNDbasis1.dat"
    A = load_matrix(filename, system)
    D = diag(A)
    N = size(A, 1)
    all_idxs = sortperm(abs.(D), rev = true)
    V0_1 = A[:, all_idxs[1:Nlow]]
    Vs = [V0_1]


    for (j, V0) in enumerate(Vs)
        println("=== Starting Davidson with initial guess V0 number $(j) ===")
        V = copy(V0)
        @time Σ, U = davidson(A, V, Naux, 1e-3, system, 200)

        idx = sortperm(Σ)
        Σ = abs.(Σ[idx])
        Σ = sqrt.(abs.(Σ))  # Take square root of eigenvalues
            
        println("Number of FLOPs: $NFLOPs")

        # Perform exact diagonalization as reference
        println("\nReading exact Eigenvalues...")
        Σexact = read_eigenresults(system)
        Σexact = abs.(Σexact)
        idx_exact = sortperm(Σexact, rev=true)
        Σexact = sqrt.(abs.(Σexact[idx_exact]))


        # Display difference
        r = length(Σ) 
        println("\nCompute the difference between computed and exact eigenvalues:")
        display("text/plain", (Σ[1:r] - Σexact[1:r])')
    end
end

systems = ["uracil"] #"uracil", , "formaldehyde", "H2"
N_lows = [10, 15, 20] # Example values for Nlow

for system in systems
    println("Running for system: $system")
    for Nlow in N_lows
        nev = Nlow * occupied_orbitals(system)
        println("Target number of eigenvalues (nev) = $nev")
        main(system, nev)  # Replace F with loop index i
    end
end