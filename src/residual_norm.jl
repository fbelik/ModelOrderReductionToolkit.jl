"""
`Affine_Residual_Init`

A struct for containing the necessary vectors
and matrices for quickly compute the `X`-norm of the
residual, `r(u_r,p) = A(p) (u - V u_r) = b(p) - A(p) V u_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`.

Here, `u` solves `A(p) u = b(p)` with `A` and `b`
having affine parameter dependence, and `V` is
a matrix with columns defining bases for approximation
spaces `u ≈ V u_r`.
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
`residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, V[, X0=nothing, T=Float64])`

Method that constructs the necessary vectors and matrices to
quickly compute the `X`-norm of the residual, 
`r(u_r,p) = A(p) (u - V u_r) = b(p) - A(p) V u_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`.

Pass as input a vector of matrices `Ais`, and a function
`makeθAi` such that the affine construction of `A` is given by
`A(p) = ∑_{i=1}^QA makeθAi(p,i) * Ais[i]`, and similarly
a vector of vectors `bis` and a function `makeθbi` such that
the affine construction of `b` is given by 
`b(p) = ∑_{i=1}^Qb makeθbi(p,i) * bis[i]`.

Additionally, pass in a matrix `V` which contains as columns
a basis for a reduced space, `u ≈ V u_r` with the dimension
of `u_r` less than that of `u`.

Optionally pass in a matrix `X` from which the `X`-norm of
the residual will be computed in the method
`residual_norm_affine_online`. If `X` remains as `nothing`,
then will choose it to be the identity matrix to compute the
2-norm of the residual.

If using complex numbers, specify `T=ComplexF64`.
"""
function residual_norm_affine_init(Ais::AbstractVector,
                                   makeθAi::Function,
                                   bis::AbstractVector,
                                   makeθbi::Function,
                                   V::AbstractVector,
                                   X::Union{Nothing,Matrix}=nothing;
                                   T::Type=Float64)
    n = length(bis[1])
    QA = length(Ais)
    Qb = length(bis)
    # Form X if nothing
    X0 = Matrix{T}(I, (n,n))
    if !isnothing(X)
        X0 .= X
    end
    # Form V as a matrix for initialization
    V_mat = length(V) == 0 ? zeros(T, (size(Ais[1],1),0)) : reduce(hcat, V)
    # Form c_ij = b_i^T X b_j
    cijs = T[]
    for i in 1:Qb
        cij = bis[i]' * X0 * bis[i]
        push!(cijs, cij)
        for j in i+1:Qb
            cij = bis[i]' * X0 * bis[j]
            push!(cijs, cij)
        end
    end
    # Form d_ij = V^T A_i^T X b_j
    dijs = Vector{T}[]
    for Ai in Ais
        for bi in bis
            dij = V_mat' * Ai' * X0 * bi
            push!(dijs, dij)
        end
    end
    # Form E_ij = V^T A_i^T X A_j V
    # Store E_ijs as vector of vectors
    Eijs = Vector{Vector{T}}[]
    for i in 1:QA
        Eii_mat = V_mat' * Ais[i]' * X0 * Ais[i] * V_mat
        Eii = eachcol(Eii_mat)#[@view Eii_mat[:,i] for i in 1:size(Eii_mat)[2]]
        push!(Eijs, Eii)
        for j in i+1:QA
            Eij_mat = V_mat' * Ais[i]' * X0 * Ais[j] * V_mat
            Eij = eachcol(Eij_mat)#[@view Eij_mat[:,i] for i in 1:size(Eij_mat)[2]]
            push!(Eijs, Eij)
        end
    end
    return Affine_Residual_Init(cijs, dijs, Eijs, Ais, makeθAi, 
                                bis, makeθbi, n, QA, Qb, V, X0)
end

"""
`add_col_to_V!(res_init, v, T)`

Method to add a vector `v` to the columns of the matrix `V`
in the `Affine_Residual_Init` object, `res_init`, without
recomputing all terms. Must specify type `T`.
"""
function add_col_to_V!(res_init::Affine_Residual_Init, v::Vector, T::Type=Float64)
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
        Eij_col = zeros(T, length(res_init.V)+1)
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
            Eij_col = zeros(T, length(res_init.V)+1)
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
`residual_norm_affine_online(res_init, u_r, p)`

Method that given `res_init`, an `Affine_Residual_Init`
object, computes the `X`-norm of the residual, 
`r(u_r,p) = A(p) (u - V u_r) = b(p) - A(p) V u_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`.

Pass as input the `Affine_Residual_Init` object, `res_init`,
a reduced vector `u_r`, and the corresponding parameter
vector `p`.
"""
function residual_norm_affine_online(res_init::Affine_Residual_Init,
                                     u_r::AbstractVector,
                                     p::AbstractVector)
    θbis = [res_init.makeθbi(p,i) for i in 1:res_init.Qb]
    θAis = [res_init.makeθAi(p,i) for i in 1:res_init.QA]
    # Sum across cijs
    res = 0.0
    idx = 1
    for i in 1:res_init.Qb
        res += real(θbis[i] * θbis[i]' * res_init.cijs[idx])
        idx += 1
        for j in i+1:res_init.Qb
            cur = θbis[i]' * θbis[j] * res_init.cijs[idx]
            res += real(cur + cur')
            idx += 1
        end
    end
    if length(res_init.V) > 0
        # Sum across dijs
        cur = 0.0
        idx = 1
        for i in 1:res_init.QA
            for j in 1:res_init.Qb
                cur -= θAis[i]' * θbis[j] * (u_r' * res_init.dijs[idx])
                idx += 1
            end
        end
        res += real(cur + cur')
        # Sum across Eijs
        idx = 1
        for i in 1:res_init.QA
            cur = 0.0
            for k in eachindex(u_r)
                cur += u_r[k] * (u_r' * res_init.Eijs[idx][k])
            end
            res += real(θAis[i] * θAis[i]' * cur)
            idx += 1
            for j in i+1:res_init.QA
                cur = 0.0
                for k in eachindex(u_r)
                    cur += u_r[k] * (u_r' * res_init.Eijs[idx][k])
                end
                cur *= θAis[i]' * θAis[j] 
                res += real(cur + cur')
                idx += 1
            end
        end
    end
    return sqrt(max(0, res))
end

mutable struct Affine_Residual_Init_Proj
    F::UpdatableQR
    qijs::Vector
    ij_idxs::Vector
    b_par_ijks::Vector
    b_perp_idxs::Vector
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

# TODO Add docstrings
function residual_norm_affine_proj_init(Ais::AbstractVector,
                                        makeθAi::Function,
                                        bis::AbstractVector,
                                        makeθbi::Function,
                                        V::AbstractVector,
                                        X::Union{Nothing,Matrix}=nothing;
                                        T::Type=Float64)
    n = length(bis[1])
    QA = length(Ais)
    Qb = length(bis)
    # Form X if nothing
    X0 = Matrix{T}(I, (n,n))
    if !isnothing(X)
        X0 .= X
    end
    # Form Updateable QR Factorization for AV
    F = UpdatableQR(zeros(T, (size(Ais[1],1),0)))
    # Store vectors of Q
    qijs = Vector{Vector{T}}[]
    ij_idxs = Vector{Int}[]
    for (i,Ai) in enumerate(Ais)
        qis = Vector{T}[]
        push!(qijs, qis)
        push!(ij_idxs, zeros(Int, length(V)))
        for (j,v) in enumerate(V)
            UpdatableQRFactorizations.add_column!(F, Ai * v)
            qij = zeros(T, F.n)
            qij[F.m] = 1
            for m in F.rot_index:-1:1
                lmul!(F.rotations_full[m]', qij)
            end
            push!(qis, qij)
            ij_idxs[i][j] = F.m
        end
    end
    # Store coefficients for b_perp
    b_perp_idxs = zeros(Int, Qb)
    for i in 1:Qb
        # Add bis to QR
        UpdatableQRFactorizations.add_column!(F, bis[i])
        b_perp_idxs[i] = F.m
    end
    # Store coefficients for b_par
    b_par_ijks = Vector{Vector{T}}[]
    for i in 1:QA
        b_par_jks = Vector{T}[]
        push!(b_par_ijks, b_par_jks)
        for j in eachindex(V)
            b_par_ks = zeros(T,Qb)
            push!(b_par_jks, b_par_ks)
            for k in 1:Qb
                b_par_ks[k] = qijs[i][j]' * bis[k]
            end
        end
    end
    return Affine_Residual_Init_Proj(F, qijs, ij_idxs, b_par_ijks, b_perp_idxs,
                                     Ais, makeθAi, bis, makeθbi, n, QA, Qb, V, X0)
end

function add_col_to_V!(res_init::Affine_Residual_Init_Proj, v::Vector, T::Type=Float64)
    QA = length(res_init.Ais)
    Qb = length(res_init.bis)
    for i in 1:Qb
        UpdatableQRFactorizations.remove_column!(res_init.F)
    end
    # Update vectors of Q
    for (i,Ai) in enumerate(res_init.Ais)
        UpdatableQRFactorizations.add_column!(res_init.F, Ai * v)
        qij = zeros(T, res_init.F.n)
        qij[res_init.F.m] = 1
        for m in res_init.F.rot_index:-1:1
            lmul!(res_init.F.rotations_full[m]', qij)
        end
        push!(res_init.qijs[i], qij)
        push!(res_init.ij_idxs[i], res_init.F.m)
    end
    push!(res_init.V, v)
    # Update indices for b_perp
    res_init.b_perp_idxs .+= QA
    # Update coefficients for b_par
    for i in 1:QA
        b_par_ks = zeros(T,Qb)
        push!(res_init.b_par_ijks[i], b_par_ks)
        for k in 1:Qb
            b_par_ks[k] = res_init.qijs[i][end]' * res_init.bis[k]
        end
    end
    for i in 1:Qb
        # Add bis to QR
        UpdatableQRFactorizations.add_column!(res_init.F, res_init.bis[i])
        res_init.b_perp_idxs[i] = res_init.F.m
    end
end

function residual_norm_affine_online(res_init::Affine_Residual_Init_Proj,
                                     u_r::AbstractVector,
                                     p::AbstractVector)
    θbis = [res_init.makeθbi(p,i) for i in 1:res_init.Qb]
    θAis = [res_init.makeθAi(p,i) for i in 1:res_init.QA]
    # Compute ||b_par(p) - A(p) V u_r||
    res = 0.0
    for i in eachindex(res_init.Ais)
        for j in eachindex(res_init.V)
            cur = 0.0
            for k in eachindex(res_init.bis)
                cur += θbis[k] * res_init.b_par_ijks[i][j][k]
            end
            R_row = res_init.ij_idxs[i][j]
            for k in eachindex(res_init.Ais)
                for l in eachindex(res_init.V)
                    R_col = res_init.ij_idxs[k][l]
                    cur -= θAis[k] * u_r[l] * res_init.F.R_full[R_row,R_col]
                end
            end
            res += cur^2
        end
    end
    # Compute ||b_perp(p)||
    for i in eachindex(res_init.bis)
        cur = 0.0
        row = res_init.b_perp_idxs[i]
        for j in i:length(res_init.bis)
            col = res_init.b_perp_idxs[j]
            cur += (θbis[j] * res_init.F.R_full[row,col])
        end
        res += cur^2
    end
    return sqrt(res)
end