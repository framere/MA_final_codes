using LinearAlgebra
using JLD2

function load_matrix(filename::String)
    N = 20000  

    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    return Hermitian(A)
end

function diagonalize_and_save(filename::String, Number::Int)    
    A = load_matrix(filename)
    println("Diagonalizing the matrix ...")
    @time F = eigen(A)  # F.values, F.vectors

    output_file = "EV_matrix_$(Number).jld2"
    println("Saving results to $output_file")

    jldsave(output_file; 
        eigenvalues = F.values, 
        eigenvectors = F.vectors
    )

    println("Done saving eigenvalues and eigenvectors.")
end

numbers = collect(6:10)
for num in numbers
    filename = "artificial_matrix_$(num).dat"
    diagonalize_and_save(filename, num)
end
