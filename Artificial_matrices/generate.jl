using LinearAlgebra
using Random

function new_curve(x, gamma, delta, epsilon)
    return gamma .* exp.(- delta .* x.^epsilon)
end

function generate_positive_definite_matrix(N::Int, factor::Int)
    A = Matrix{Float64}(undef, N, N)

    # Your parameters
    a = 31.2458
    b = 0.647351
    c = 0.355697

    # Fill matrix
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = new_curve(i, a, b, c)
            else
                rand_val = rand()
                if rand_val < 0.95
                    A[i,j] = new_curve(abs(i - j), a, b, c) / factor
                else
                    A[i,j] = new_curve(abs(i - j), a, b, c) * rand()
                end
            end
        end
    end

    # Make the matrix Hermitian
    A = (A + A') / 2

    # Make the matrix positive definite
    # Robust approach: A' * A ensures PSD, then add small shift
    # A = A' * A
    A += 1e-9 * I

    return A
end


function save_matrix_to_file(A::Matrix{Float64}, filename::String)
    """    save_matrix_to_file(A::Matrix{Float64}, filename::String)
    Saves the Hermitian matrix `A` to a file in a flattened format.
    """
    A_vec = vec(Matrix(A))  # Flatten to 1D
    open(filename, "w") do file
        write(file, A_vec)
    end
end

# === MAIN USAGE ===
Ns = [20000]  # Different sizes for testing
factors = [10^4]  # [1e8, 1e2]
# factors = 10^12

global counter = 9
for (N, factor) in Iterators.product(Ns, factors)
    println("Generating a random Hermitian matrix of size $N x $N with diagonal scaling factor $factor...")
    A = generate_positive_definite_matrix(N, factor)
    filename = "artificial_matrix_$(counter).dat"
    println("Saving the generated matrix to '$filename'...")
    save_matrix_to_file(A, filename)
    global counter += 1
end
