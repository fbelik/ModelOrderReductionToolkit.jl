"""
Find a low rank solution for the generalized continuous-time Lyapunov equation 
`AXE' + EXA' = -FF'`. Returns a matrix `Z` such that `X ≈ ZZ'`.
"""
function glyap_lradi_r(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat; eps=1e-2, maxdim=-1, noise=1)
    W = copy(B)
    E = isa(E, UniformScaling) ? Diagonal(E.λ * ones(size(A, 2))) : E
    Z = VectorOfVectors(size(A, 2), 0, Float64)
    Z_ortho = VectorOfVectors(size(A, 2), 0, Float64)
    ZoAZo = VectorOfVectors(0, 0, Float64)
    ZoEZo = VectorOfVectors(0, 0, Float64)
    shifts = init_projection_shifts(A, E, B, noise=noise)
    j = 1
    j_shift = 1
    if maxdim == -1
        maxdim = size(A, 2)
    end
    while true
        α = shifts[j_shift]
        j_shift += 1
        if imag(α) == 0
            V = (A .+ (real(α) .* E)) \ W
            W -= (2 * real(α)) * (E * V)
            Z_add = sqrt(-2 * real(α)) * V
            for z in eachcol(Z_add)
                addCol!(Z, z)
                zn = copy(z)
                orthonormalize_mgs2!(zn, Z_ortho)
                addCol!(Z_ortho, zn)
                j += 1
            end
        else
            V = (A .+ (α .* E)) \ W
            γ = 2 * sqrt(-1 * real(α))
            δ = real(α) / imag(α)
            reV = real.(V) .+ δ * imag.(V)
            W += γ^2 * (E * reV)
            Z_add = γ * reV
            for z in eachcol(Z_add)
                addCol!(Z, z)
                zn = copy(z)
                orthonormalize_mgs2!(zn, Z_ortho)
                addCol!(Z_ortho, zn)
                j += 1
            end
            Z_add = γ * sqrt(δ^2 + 1) * imag.(V)
            for z in eachcol(Z_add)
                addCol!(Z, z)
                zn = copy(z)
                orthonormalize_mgs2!(zn, Z_ortho)
                addCol!(Z_ortho, zn)
                j += 1
            end
        end
        if j_shift > length(shifts)
            shifts = more_projection_shifts(A, E, V, Z_ortho, shifts, noise=noise)
            j_shift = 1
        end
        errval = norm(W'W)
        if errval < eps
            if noise >= 1
                println("Norm of gram of W less than ϵ, returning at dimension $(j-1)")
            end
            break
        elseif j > maxdim
            if noise >= 1
                println("Reached maximum dimension without hitting tolerance, returning")
            end
            break
        elseif noise >= 2
            @printf("(%d) Norm of gram %.3e\n", j-1, errval)
        end
    end
    return Matrix(Z)
end

function init_projection_shifts(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat; noise=1)
    Q = Matrix(qr(B).Q)
    QAQ = Q'*A*Q
    QEQ = isa(E, UniformScaling) ? Diagonal(E.λ .* ones(size(QAQ, 2))) : Q'*E*Q
    shifts = eigen(QAQ, QEQ).values
    shifts = [s for s in shifts if real(s) < 0]
    if length(shifts) == 0
        if noise >= 1
            println("Zero init shifts found, using random subspace of same dimension")
        end
        B = rand(size(B))
        Q = Matrix(qr(B).Q)
        QAQ = Q'*A*Q
        QEQ = isa(E, UniformScaling) ? Diagonal(E.λ .* ones(size(A, 2))) : Q'*E*Q
        shifts = eigen(QAQ, QEQ).values
        shifts = [s for s in shifts if real(s) < 0]
        if length(shifts) == 0
            error("Unable to initialize shifts")
        end
    end
    return shifts
end

function more_projection_shifts(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, V::AbstractVecOrMat, Z_ortho::AbstractMatrix, prev_shifts::AbstractVector; noise=1)
    Q = isa(Z_ortho, VectorOfVectors) ? Matrix(Z_ortho) : Z_ortho
    QAQ = Q'*A*Q
    QEQ = isa(E, UniformScaling) ? Diagonal(E.λ .* ones(size(QAQ, 2))) : Q'*E*Q
    shifts = eigen(QAQ, QEQ).values
    shifts = [s for s in shifts if real(s) < 0 &&  imag(s) >= 0]
    if length(shifts) == 0
        if noise >= 1
            println("Zero new shifts found, returning previous shifts")
        end
        return prev_shifts
    end
    return shifts
end