"""
`singular_values_information(σs)`

Given a vector of real singular values `σs`, return a vector
providing the maximum percentage of "information" available
from a `d`-dimensional representation in Frobenius norm.

Uses the fact that the maximum error in projecting onto the first `d`
left singular vectors is given by

`||A-A_d||_F = √ (σ_{d+1}^2 + ... + σ_N^2)`.

So the information with a `d`-dimensional representation given by

`singular_values_information[d] = 1 - √ (σ_{d+1}^2 + ... + σ_N^2) / √ (σ_1^2 + ... + σ_N^2)`
"""
function singular_values_information(σs::AbstractVector) :: Vector{Float64}
    N = length(σs)
    res = zeros(N)
    for d in N-1:-1:1
        # res[d] = σ_{d+1}^2 + ... + σ_N^2
        res[d] = σs[d+1] ^ 2 + res[d+1]
    end
    # Sum of squares of singular values
    sumsq = σs[1] ^ 2 + res[1]
    res .= 1 .- sqrt.(res ./ sumsq)
    return res
end

"""
`P = pca_projector(A, dim=-1; info_tol=1e-2)`

Uses the first dim singular vectors of `A` to
generate an orthogonal projection operator `P`.
Operator performs matrix operation and can be
done on vectors or matrices of proper dimension.

If `dim` is not provided, uses `n` singular values 
where `n` is the smallest integer such that the amount
of information per the Frobenious norm is at least
`1-info_tol`.

Returns a functor `P` which stores the reduced dimension
`n`, the dimension of vectors to project `m`, the singular
values of `A` in `s`, and the vectors to project onto
in the `m × n` matrix `M`.

Call with `P(b)` where `b` is a `m` vector or a matrix
with `m` rows, projecting each column.
"""
function pca_projector(
    A::AbstractMatrix, 
    dim::Int=-1;
    info_tol::Float64=1e-2) :: Function

    # Perform SVD and obtain left singular vectors and singular values
    U,s,_ = LinearAlgebra.svd(A)
    if dim == -1
        svi = singular_values_information(s)
        n0 = findfirst(x -> x >= (1-info_tol), svi)
    else
        n0 = dim
    end
    M = U[:,1:n0]
    m = size(A)[1]
    n = n0 # Avoid type Core.Box
    # Orthogonal projector
    P(b::AbstractVecOrMat, full::Bool=true) = begin
        # Variables to keep in scope
        n; m; s
        return full ? M * (M' * b) : M' * b 
    end
    return P
end

"""
`qr_projector(A, dim)`

Uses the first `dim` columns of `A` by qr-pivot order
(norm) to generate an orthogonal projection operator `P`.
Operator performs matrix operation and can be
done on vectors or matrices of proper dimension.

Returns a functor `P` which stores the reduced dimension
`n`, the dimension of vectors to project `m`, the vectors 
to project onto in the `m × n` matrix `M`, and the matrix
`inv(M'M)` in `MtMinv`.

Call with `P(b)` where `b` is a `m` vector or a matrix
with `m` rows, projecting each column.
"""
function qr_projector(
    A::AbstractMatrix, 
    dim::Int) :: Function
    
    # Perform QR with pivots and obtain pivoted columns
    Q,_,_ = LinearAlgebra.qr(A, LinearAlgebra.ColumnNorm())
    M = Q[:,1:dim]
    m = size(M)[1]
    n = dim
    # Orthogonal projector
    P(b::AbstractVecOrMat, full::Bool=true) = begin 
        # Variables to keep in scope
        n; m
        return full ? M * (M' * b) : M' * b
    end
    return P
end

"""
`P = eim_projector(A, dim)`

Uses discrete empirical interpolation to generate a projector
operator `P` that selects a few rows from the given vector or
matrix, and then projects them onto a set of columns chosen
by a greedy procedure.

Returns a functor `P` which stores the reduced dimension
`n`, the dimension of vectors to project `m`, the matrix
`A_C` which is `A` with `n` extracted columns, and the matrix
`A_CR` which is `A` with `n` extracted rows and columns.

Call with `P(b)` where `b` is a `m` vector or a matrix
with `m` rows, projecting each column.
"""
function eim_projector(
    A::AbstractMatrix, 
    dim::Int) :: Function

    # Perform LU
    p,_,_,q = full_lu(A, steps=dim)
    A_C = A[:,q[1:dim]]
    A_CR = A_C[p[1:dim],:]
    # Orthogonal projector
    m = size(A)[1]
    n = dim
    P(b::AbstractVecOrMat, full::Bool=true) = begin 
        # Variables to keep in scope
        n; m
        return full ? A_C * (A_CR \ b[p[1:n],:]) : A_CR \ b[p[1:n],:]
    end
    return P
end