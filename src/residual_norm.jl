using LinearAlgebra
using StaticArrays

"""
`Affine_Residual_Init`

A struct for containing the necessary vectors
and matrices for quickly compute the `X`-norm of the
residual, `r(x_r,p) = A(p) (x - V x_r) = b(p) - A(p) V x_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`.

Here, `x` solves `A(p) x = b(p)` with `A` and `b`
having affine parameter dependence, and `V` is
a matrix with columns defining bases for approximation
spaces `x ≈ V x_r`.
"""
mutable struct Affine_Residual_Init
    cijs::Vector
    dijs::Vector
    Eijs::Vector
    Ais::Vector
    makeθAi::Function
    bis::Vector
    makeθbi::Function
    n::Int
    QA::Int
    Qb::Int
    V::Vector
    X::Matrix
end

"""
`residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, V[, X0=nothing])`

Method that constructs the necessary vectors and matrices to
quickly compute the `X`-norm of the residual, 
`r(x_r,p) = A(p) (x - V x_r) = b(p) - A(p) V x_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`.

Pass as input a vector of matrices `Ais`, and a function
`makeθAi` such that the affine construction of `A` is given by
`A(p) = ∑_{i=1}^QA makeθAi(p,i) * Ais[i]`, and similarly
a vector of vectors `bis` and a function `makeθbi` such that
the affine construction of `b` is given by 
`b(p) = ∑_{i=1}^Qb makeθbi(p,i) * bis[i]`.

Additionally, pass in a matrix `V` which contains as columns
a basis for a reduced space, `x ≈ V x_r` with the dimension
of `x_r` less than that of `x`.

Optionally pass in a matrix `X` from which the `X`-norm of
the residual will be computed in the method
`residual_norm_affine_online`. If `X` remains as `nothing`,
then will choose it to be the identity matrix to compute the
2-norm of the residual.
"""
function residual_norm_affine_init(Ais::AbstractVector,
                                   makeθAi::Function,
                                   bis::AbstractVector,
                                   makeθbi::Function,
                                   V::AbstractMatrix,
                                   X::Union{Nothing,Matrix}=nothing)
    n = length(bis[1])
    QA = length(Ais)
    Qb = length(bis)
    # Form X if nothing
    X0 = Matrix{Float64}(I, (n,n))
    if !isnothing(X)
        X0 .= X
    end
    # Form c_ij = b_i^T X b_j
    cijs = Float64[]
    for i in 1:Qb
        cij = bis[i]' * X0 * bis[i]
        push!(cijs, cij)
        for j in i+1:Qb
            cij = bis[i]' * X0 * bis[j]
            push!(cijs, cij)
        end
    end
    # Form d_ij = V^T A_i^T X b_j
    dijs = Vector{Float64}[]
    for Ai in Ais
        for bi in bis
            dij = V' * Ai' * X0 * bi
            push!(dijs, dij)
        end
    end
    # Form E_ij = V^T A_i^T X A_j V
    # Store E_ijs as vector of vectors
    Eijs = Vector{Vector{Float64}}[]
    for i in 1:QA
        Eii_mat = V' * Ais[i]' * X0 * Ais[i] * V
        Eii = [@view Eii_mat[:,i] for i in 1:size(Eii_mat)[2]]
        push!(Eijs, Eii)
        for j in i+1:QA
            Eij_mat = V' * Ais[i]' * X0 * Ais[j] * V
            Eij = [@view Eij_mat[:,i] for i in 1:size(Eij_mat)[2]]
            push!(Eijs, Eij)
        end
    end
    # Store V as vector of vectors
    V = Vector{Float64}[V[:,i] for i in 1:size(V)[2]]
    return Affine_Residual_Init(cijs, dijs, Eijs, Ais, makeθAi, 
                                bis, makeθbi, n, QA, Qb, V, X0)
end

"""
`add_col_to_V(res_init, v)`

Method to add a vector `v` to the columns of the matrix `V`
in the `Affine_Residual_Init` object, `res_init`, without
recomputing all terms.
"""
function add_col_to_V(res_init::Affine_Residual_Init, v::Vector)
    # Add to end of each dij vector
    idx = 1
    for Ai in res_init.Ais
        for bi in res_init.bis
            dij_end = v' * Ai' * res_init.X * bi
            push!(res_init.dijs[idx], dij_end)
            idx += 1
        end
    end
    # Append to Eij matrices
    idx = 1
    for i in eachindex(res_init.Ais)
        # Add to each row of V
        for k in 1:length(res_init.V)
            newrowval = v' * res_init.Ais[i]' * res_init.X' * res_init.Ais[i] * res_init.V[k]
            push!(res_init.Eijs[idx][k], newrowval)
        end
        # Form new column of V
        Eij_col = zeros(length(res_init.V)+1)
        for k in 1:length(res_init.V)
            newcolval = res_init.V[k]' * res_init.Ais[i]' * res_init.X' * res_init.Ais[i] * v
            Eij_col[k] = newcolval
        end
        Eij_col[end] = v' * res_init.Ais[i]' * res_init.X' * res_init.Ais[i] * v
        push!(res_init.Eijs[idx], Eij_col)
        idx += 1
        for j in i+1:length(res_init.Ais)
            # Add to each row of V
            for k in 1:length(res_init.V)
                newrowval = v' * res_init.Ais[i]' * res_init.X' * res_init.Ais[j] * res_init.V[k]
                push!(res_init.Eijs[idx][k], newrowval)
            end
            # Form new column of V
            Eij_col = zeros(length(res_init.V)+1)
            for k in 1:length(res_init.V)
                newcolval = res_init.V[k]' * res_init.Ais[i]' * res_init.X' * res_init.Ais[j] * v
                Eij_col[k] = newcolval
            end
            Eij_col[end] = v' * res_init.Ais[i]' * res_init.X' * res_init.Ais[j] * v
            push!(res_init.Eijs[idx], Eij_col)
            idx += 1
        end
    end
    # Append to V matrix
    push!(res_init.V, v)
end

"""
`residual_norm_affine_online(res_init, x_r, p)`

Method that given `res_init`, an `Affine_Residual_Init`
object, computes the `X`-norm of the residual, 
`r(x_r,p) = A(p) (x - V x_r) = b(p) - A(p) V x_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`.

Pass as input the `Affine_Residual_Init` object, `res_init`,
a reduced vector `x_r`, and the corresponding parameter
vector `p`.
"""
function residual_norm_affine_online(res_init::Affine_Residual_Init,
                                     x_r::AbstractVector,
                                     p::AbstractVector)
    θbis = [res_init.makeθbi(p,i) for i in 1:res_init.Qb]
    θAis = [res_init.makeθAi(p,i) for i in 1:res_init.QA]
    # Sum across cijs
    res = 0.0
    idx = 1
    for i in 1:res_init.Qb
        res += θbis[i]^2 * res_init.cijs[idx]
        idx += 1
        for j in i+1:res_init.Qb
            res += 2 * θbis[i] * θbis[j] * res_init.cijs[idx]
            idx += 1
        end
    end
    # Sum across dijs
    idx = 1
    for i in 1:res_init.QA
        for j in 1:res_init.Qb
            res += -2 * θAis[i] * θbis[j] * (x_r' * res_init.dijs[idx])
            idx += 1
        end
    end
    # Sum across Eijs
    idx = 1
    for i in 1:res_init.QA
        cur = 0.0
        for k in eachindex(x_r)
            cur += x_r[k] * dot(res_init.Eijs[idx][k], x_r)
        end
        res += θAis[i]^2 * cur
        idx += 1
        for j in i+1:res_init.QA
            cur = 0.0
            for k in eachindex(x_r)
                cur += x_r[k] * dot(res_init.Eijs[idx][k], x_r)
            end
            res += 2 * θAis[i] * θAis[j] * cur
            idx += 1
        end
    end
    return sqrt(max(0, res))
end

"""
`residual_norm_explicit(x_approx, p, makeA, makeb, X=nothing)`

Method that computes the `X`-norm of the residual, 
`r(x_r,p) = A(p) (x - x_approx) = b(p) - A(p) x_approx` explicitly
where `x` is the solution to `A(p) x = b(p)`, and `x_approx` is
an approximation, `x ≈ x_approx`.

Pass as input the approximation vector `x_approx`, the
corresponding parameter, `p`, the method `makeA(p)` for
constructing the matrix `A(p)`, and the method `makeb(p)`
for constructing the vector `b(p)`.

Optionally pass in a matrix `X` from which the `X`-norm of
the residual will be computed. If `X` remains as `nothing`,
then will choose it to be the identity matrix to compute the
2-norm of the residual.
"""
function residual_norm_explicit(x_approx::AbstractVector,
                                p::AbstractVector,
                                makeA::Function,
                                makeb::Function,
                                X::Union{Nothing,Matrix}=nothing)
    # Residual defined to be A(p)(x-Vx_r) = b(p) - A(p)Vx_r
    b = makeb(p)
    A = makeA(p)
    r = b .- A*x_approx
    if isnothing(X)
        return sqrt(r'r)
    else
        return sqrt(r'*X*r)
    end
end

"""
`residual_norm_explicit(x_approx, p, Ais, makeθAi, bis, makeθbi, X=nothing)`

Method that computes the `X`-norm of the residual, 
`r(x_r,p) = A(p) (x - x_approx) = b(p) - A(p) x_approx` explicitly
where `x` is the solution to `A(p) x = b(p)`, and `x_approx` is
an approximation, `x ≈ x_approx`.

Pass as input the approximation vector `x_approx`, the
corresponding parameter, `p`, a vector of matrices `Ais`, 
and a function `makeθAi` such that the affine construction 
of `A` is given by `A(p) = ∑_{i=1}^QA makeθAi(p,i) * Ais[i]`, 
and similarly a vector of vectors `bis` and a function `makeθbi` 
such that the affine construction of `b` is given by 
`b(p) = ∑_{i=1}^Qb makeθbi(p,i) * bis[i]`.

Optionally pass in a matrix `X` from which the `X`-norm of
the residual will be computed. If `X` remains as `nothing`,
then will choose it to be the identity matrix to compute the
2-norm of the residual.
"""
function residual_norm_explicit(x_approx::AbstractVector,
                                p::AbstractVector,
                                Ais::AbstractVector,
                                makeθAi::Function,
                                bis::AbstractVector,
                                makeθbi::Function,
                                X::Union{Nothing,Matrix}=nothing)
    makeA(p) = begin
        A = zeros(size(Ais[1]))
        for i in eachindex(Ais)
            A .+= makeθAi(p,i) .* Ais[i]
        end
        A
    end
    makeb(p) = begin
        b = zeros(length(bis[1]))
        for i in eachindex(bis)
            b .+= makeθbi(p,i) .* bis[i]
        end
        b
    end
    return residual_norm_explicit(x_approx, p, makeA, makeb, X)
end