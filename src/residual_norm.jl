"""
`res <: ResidualNormComputer{T}`

Abstract type for computing the norm of the residual
for weak greedy RB methods. Must implement the
`update!(res, v)` and `compute(res, u_r, p)` methods.
"""
abstract type ResidualNormComputer{T} end

"""
`update!(res, v)`

Method to add a vector `v` to the ResidualNormComputer
`res` and update internals.
"""
function update!(res::ResidualNormComputer, v::AbstractVector)
    error("Must implement update!(res,v) for res <: ResidualNormComputer")
end

"""
`compute(res, u_r, p)`

Method to compute the residual norm ||b(p) - A(p) V u_r|| for the
ResidualNormComputer `res`.
"""
function compute(res_init::ResidualNormComputer, u_r::AbstractVector, p)
    error("Must implement compute(res, u_r, p) for res <: ResidualNormComputer")
end

"""
`StandardResidualNormComputer{T} <: ResidualNormComputer{T}`
`StandardResidualNormComputer(Ap::APArray,bp::APArray,V=nothing,X=nothing)`

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
struct StandardResidualNormComputer{T} <: ResidualNormComputer{T}
    cijs::Vector{T}
    dijs::Vector{Vector{T}}
    Eijs::Vector{VectorOfVectors{T}}
    Ap::APArray
    bp::APArray
    QA::Int
    Qb::Int
    V::VectorOfVectors{T}
    X::Union{AbstractMatrix,UniformScaling}
end

function StandardResidualNormComputer(Ap::APArray,bp::APArray,V::Union{VectorOfVectors,Nothing}=nothing,X::Union{Nothing,AbstractMatrix,UniformScaling}=nothing)
    if isnothing(V)
        T1 = typeof(prod(zero.(eltype.(Ap.arrays))))
        T2 = typeof(prod(zero.(eltype.(bp.arrays))))
        T = typeof(zero(T1) * zero(T2))
        V = VectorOfVectors(size(Ap.arrays[1], 1), 0, T)
    else
        T = eltype(V)
    end
    Ais = Ap.arrays
    makeθAi = Ap.makeθi
    bis = bp.arrays
    makeθbi = bp.makeθi

    n = length(bis[1])
    QA = length(Ais)
    Qb = length(bis)
    # Form X if nothing
    X0 = isnothing(X) ? I : X
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
            dij = V' * (Ai' * X0 * bi)
            push!(dijs, dij)
        end
    end
    # Form E_ij = V^T A_i^T X A_j V
    # Store E_ijs as vector of vectors
    Eijs = VectorOfVectors{T}[]
    for i in 1:QA
        Eii_mat = V' * Ais[i]' * X0 * Ais[i] * V
        push!(Eijs, VOV(Eii_mat))
        for j in i+1:QA
            Eij_mat = V' * Ais[i]' * X0 * Ais[j] * V
            push!(Eijs, VOV(Eij_mat))
        end
    end
    return StandardResidualNormComputer{T}(cijs, dijs, Eijs, Ap, bp, QA, Qb, V, X0)
end

function update!(res_init::StandardResidualNormComputer, v::AbstractVector)
    Ais = res_init.Ap.arrays
    bis = res_init.bp.arrays
    # Add to end of each dij vector
    idx = 1
    for Ai in Ais
        for bi in bis
            dij_end = v' * Ai' * res_init.X * bi
            push!(res_init.dijs[idx], dij_end)
            idx += 1
        end
    end
    # Append to Eij matrices
    idx = 1
    for i in 1:res_init.QA
        # Add to each row of V
        addRow!(res_init.Eijs[idx])
        res_init.Eijs[idx][end:end,:] .= (((v' * Ais[i]') * res_init.X') * Ais[i]) * res_init.V
        # Form new column of V
        addCol!(res_init.Eijs[idx])
        res_init.Eijs[idx][1:end-1,end:end] .= ((((v' * Ais[i]') * res_init.X) * Ais[i]) * res_init.V)'
        res_init.Eijs[idx][end,end] = (((v' * Ais[i]') * res_init.X') * Ais[i]) * v
        idx += 1
        for j in (i+1):res_init.QA
            # Add to each row of V
            addRow!(res_init.Eijs[idx])
            res_init.Eijs[idx][end:end,:] .= (((v' * Ais[i]') * res_init.X') * Ais[j]) * res_init.V
            # Form new column of V
            addCol!(res_init.Eijs[idx])
            res_init.Eijs[idx][1:end-1,end:end] .= ((((v' * Ais[j]') * res_init.X) * Ais[i]) * res_init.V)'
            res_init.Eijs[idx][end,end] = (((v' * Ais[i]') * res_init.X') * Ais[j]) * v
            idx += 1
        end
    end
    # Append column to V
    addCol!(res_init.V, v)
end

function compute(res_init::StandardResidualNormComputer, u_r::AbstractVector, p)
    θbis = res_init.bp.precompθ ? res_init.bp.makeθ(p) : [res_init.bp.makeθi(p,i) for i in 1:res_init.Qb]
    θAis = res_init.Ap.precompθ ? res_init.Ap.makeθ(p) : [res_init.Ap.makeθi(p,i) for i in 1:res_init.QA]
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
    if size(res_init.V, 2) > 0
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
            cur = u_r' * (res_init.Eijs[idx] * u_r)
            res += real(θAis[i] * θAis[i]' * cur)
            idx += 1
            for j in i+1:res_init.QA
                cur = u_r' * (res_init.Eijs[idx] * u_r)
                cur *= θAis[i]' * θAis[j] 
                res += real(cur + cur')
                idx += 1
            end
        end
    end
    return sqrt(max(0, res))
end

"""
`ProjectionResidualNormComputer{T} <: ResidualNormComputer{T}`
`ProjectionResidualNormComputer(Ap::APArray,bp::APArray,V=nothing,X=nothing)`

A struct for containing the necessary vectors
and matrices for quickly compute the `X`-norm of the
residual, `r(u_r,p) = A(p) (u - V u_r) = b(p) - A(p) V u_r`,
by taking advantage of affine parameter dependence of `A(p)`
and `b(p)`. Uses a projection method which is more
stable than the standard method.

Here, `u` solves `A(p) u = b(p)` with `A` and `b`
having affine parameter dependence, and `V` is
a matrix with columns defining bases for approximation
spaces `u ≈ V u_r`.
"""
struct ProjectionResidualNormComputer{T} <: ResidualNormComputer{T}
    F::UpdatableQR
    qijs::Vector{Vector{Vector{T}}}
    ij_idxs::Vector{Vector{Int}}
    b_par_ijks::Vector{Vector{Vector{T}}}
    b_perp_idxs::Vector{Int}
    Ap::APArray
    bp::APArray
    QA::Int
    Qb::Int
    V::VectorOfVectors{T}
    X::Union{AbstractMatrix,UniformScaling}
end

function ProjectionResidualNormComputer(Ap::APArray,bp::APArray,V::Union{VectorOfVectors,Nothing}=nothing,X::Union{Nothing,AbstractMatrix,UniformScaling}=nothing)
    if isnothing(V)
        T1 = typeof(prod(zero.(eltype.(Ap.arrays))))
        T2 = typeof(prod(zero.(eltype.(bp.arrays))))
        T = typeof(zero(T1) * zero(T2))
        V = VectorOfVectors(size(Ap.arrays[1], 1), 0, T)
    else
        T = eltype(V)
    end
    
    Ais = Ap.arrays
    bis = bp.arrays

    n = length(bis[1])
    QA = length(Ais)
    Qb = length(bis)
    # Form X if nothing
    X0 = isnothing(X) ? I : X
    # Form Updateable QR Factorization for AV
    F = UpdatableQR(zeros(T, (size(Ais[1],1),0)))
    # Store vectors of Q
    qijs = Vector{Vector{T}}[]
    ij_idxs = Vector{Int}[]
    for (i,Ai) in enumerate(Ais)
        qis = Vector{T}[]
        push!(qijs, qis)
        push!(ij_idxs, zeros(Int, length(V)))
        for (j,v) in enumerate(eachcol(V))
            if F.m < F.n
                UpdatableQRFactorizations.add_column!(F, Ai * v)
                qij = zeros(T, F.n)
                qij[F.m] = 1
                for m in F.rot_index:-1:1
                    lmul!(F.rotations_full[m]', qij)
                end
                push!(qis, qij)
                ij_idxs[i][j] = F.m
            else
                ij_idxs[i][j] = -1
            end
        end
    end
    # Store coefficients for b_perp
    b_perp_idxs = zeros(Int, Qb)
    for i in 1:Qb
        # Add bis to QR
        if F.m < F.n
            UpdatableQRFactorizations.add_column!(F, bis[i])
            b_perp_idxs[i] = F.m
        else
            b_perp_idxs[i] = -1
        end
    end
    # Store coefficients for b_par
    b_par_ijks = Vector{Vector{T}}[]
    for i in 1:QA
        b_par_jks = Vector{T}[]
        push!(b_par_ijks, b_par_jks)
        for j in eachindex(V)
            if (ij_idxs[i][j]) != -1
                b_par_ks = zeros(T,Qb)
                push!(b_par_jks, b_par_ks)
                for k in 1:Qb
                    b_par_ks[k] = qijs[i][j]' * bis[k]
                end
            end
        end
    end
    return ProjectionResidualNormComputer{T}(F, qijs, ij_idxs, b_par_ijks, b_perp_idxs,
                                        Ap, bp, QA, Qb, V, X0)
end

function update!(res_init::ProjectionResidualNormComputer{T}, v::AbstractVector{T}) where T
    Ais = res_init.Ap.arrays
    bis = res_init.bp.arrays

    for i in 1:res_init.Qb
        if res_init.b_perp_idxs[i] != -1
            UpdatableQRFactorizations.remove_column!(res_init.F)
        end
    end
    # Update vectors of Q
    for (i,Ai) in enumerate(Ais)
        if res_init.F.m < res_init.F.n
            UpdatableQRFactorizations.add_column!(res_init.F, Ai * v)
            qij = zeros(T, res_init.F.n)
            qij[res_init.F.m] = 1
            for m in res_init.F.rot_index:-1:1
                lmul!(res_init.F.rotations_full[m]', qij)
            end
            push!(res_init.qijs[i], qij)
            push!(res_init.ij_idxs[i], res_init.F.m)
        else
            if res_init.ij_idxs[i][end] != -1
                push!(res_init.ij_idxs[i], -1)
            end
        end
    end
    addCol!(res_init.V, v)
    # Update indices for b_perp
    res_init.b_perp_idxs .+= res_init.QA
    # Update coefficients for b_par
    for i in 1:res_init.QA
        if (res_init.ij_idxs[i][end]) != -1
            b_par_ks = zeros(T,res_init.Qb)
            push!(res_init.b_par_ijks[i], b_par_ks)
            for k in 1:res_init.Qb
                b_par_ks[k] = res_init.qijs[i][end]' * bis[k]
            end
        end
    end
    for i in 1:res_init.Qb
        # Add bis to QR
        if res_init.F.m < res_init.F.n
            UpdatableQRFactorizations.add_column!(res_init.F, bis[i])
            res_init.b_perp_idxs[i] = res_init.F.m
        else
            res_init.b_perp_idxs[i] = -1
        end
    end
end

function compute(res_init::ProjectionResidualNormComputer, u_r::AbstractVector, p)
    θbis = res_init.bp.precompθ ? res_init.bp.makeθ(p) : [res_init.bp.makeθi(p,i) for i in 1:res_init.Qb]
    θAis = res_init.Ap.precompθ ? res_init.Ap.makeθ(p) : [res_init.Ap.makeθi(p,i) for i in 1:res_init.QA]
    # Compute ||b_par(p) - A(p) V u_r||
    res = 0.0
    for i in 1:res_init.QA
        for j in eachindex(eachcol(res_init.V))
            if res_init.ij_idxs[i][min(j, length(res_init.ij_idxs[i]))] != -1
                cur = 0.0
                for k in 1:res_init.Qb
                    cur += θbis[k] * res_init.b_par_ijks[i][j][k]
                end
                R_row = res_init.ij_idxs[i][min(j, length(res_init.ij_idxs[i]))]
                if R_row != -1
                    for k in 1:res_init.QA
                        for l in eachindex(eachcol(res_init.V))
                            R_col = res_init.ij_idxs[k][min(l, length(res_init.ij_idxs[k]))]
                            if R_col != -1
                                cur -= θAis[k] * u_r[l] * res_init.F.R_full[R_row,R_col]
                            end
                        end
                    end
                end
                res += real(cur' * cur)
            end
        end
    end
    # Compute ||b_perp(p)||
    for i in 1:res_init.Qb
        cur = 0.0
        row = res_init.b_perp_idxs[i]
        if row != -1
            for j in i:res_init.Qb
                col = res_init.b_perp_idxs[j]
                if col != -1
                    cur += (θbis[j] * res_init.F.R_full[row,col])
                end
            end
        end
        res += real(cur' * cur)
    end
    return sqrt(res)
end