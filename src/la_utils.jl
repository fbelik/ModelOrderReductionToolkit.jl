"""
`full_lu(A; steps=-1)`

Performs a completely pivoted LU factorization
on the matrix `A`, returning permutation vectors `Q`
and `P`, and lower and upper triangular matrices `L`
and `U`, such that if all steps are performed, then
`A[P,Q] = L*U`.
"""
function full_lu(A::AbstractMatrix;steps::Int=-1)
    sA = size(A)
    if steps == -1
        steps = minimum(sA)
    end
    P = Vector(1:sA[1])
    Q = Vector(1:sA[2])
    L = zeros(sA[1],sA[1])
    for i in 1:sA[1]
        L[i,i] = 1
    end
    U = copy(A)
    tmp = zeros(maximum(sA))
    for s in 1:steps
        maxEl = 0
        maxIdx = [-1,-1]
        for j in s:sA[2], i in s:sA[1]
            absEl = abs(U[i,j])
            if absEl > maxEl
                maxIdx[1] = i
                maxIdx[2] = j
                maxEl = absEl
            end
        end
        # Perform pivoting
        P[s],P[maxIdx[1]] = P[maxIdx[1]],P[s]
        Q[s],Q[maxIdx[2]] = Q[maxIdx[2]],Q[s]
        tmp[1:sA[2]] .= U[s,:]
        U[s,:] .= U[maxIdx[1],:]
        U[maxIdx[1],:] .= tmp[1:sA[2]]
        tmp[1:sA[1]] .= U[:,s]
        U[:,s] .= U[:,maxIdx[2]]
        U[:,maxIdx[2]] .= tmp[1:sA[1]]
        tmp[1:s-1] .= L[s,1:s-1]
        L[s,1:s-1] .= L[maxIdx[1],1:s-1]
        L[maxIdx[1],1:s-1] .= tmp[1:s-1]
        # Perform Gaussian Elimination
        for i in s+1:sA[1]
            L[i,s] = U[i,s] / U[s,s]
            U[i,:] .= U[i,:] .- L[i,s] .* U[s,:]
        end
    end
    return (P,L,U,Q)
end

"""
`smallest_real_eigval(A, kmaxiter[, noise=1, krylovsteps=10])`

Given a hermitian matrix `A`, attempts to compute the 
most negative (real) eigenvalue. First, uses Krylov iteration
with a shift-invert procedure with Gershgorin disks, and if 
not successful, calls a full, dense, eigensolve.
"""
function smallest_real_eigval(A::AbstractMatrix, kmaxiter, noise=1, krylovsteps=10, shifttol=1e4)
    # Try invert method per Gershgorin disks
    mingd = 0.0
    mingd_center = 0.0
    for i in 1:size(A)[1]
        newmingd = real(A[i,i]) - (sum(abs.(view(A,i,:))) - abs(A[i,i]))
        if newmingd < mingd
            mingd = newmingd
            mingd_center = real(A[i,i])
        end
    end

    log_tilde(x) = x >= 1 ? log(x) : (x <= -1 ? -log(-x) : 0)
    exp_tilde(x) = x > 0 ? exp(x) : -exp(-x)
    exprange = exp_tilde.(range(log_tilde(mingd), log_tilde(mingd_center), krylovsteps))
    for (i,sigma) in enumerate(exprange[1:end-1])
        shift_invert = (x -> (A - sigma*I) \ x) 
        res = eigsolve(shift_invert, size(A,1), 1, :LM, eltype(A), ishermitian=true)#, maxiter=kmaxiter, krylovdim=size(A,1))
        resval = 0.0
        # Pluck the maximum result (or maximum negative if negative values exist)
        for val in res[1]
            if val < 0 && resval >= 0
                resval = val
            elseif val < 0 # && resval < 0
                resval = max(val, resval)
            elseif resval >= 0 # && val >= 0
                resval = max(val, resval)
            end # ignore if resval < 0 && val >= 0
        end
        # Continue to iterate until converged and distance from sigma smaller than next step
        if res[3].converged >= 1 && (1 / resval) < shifttol
            return sigma + 1 / resval
        end
    end
    if noise >= 1
        println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
    end
    # Perform brute eigen
    res = eigen!(issparse(A) ? collect(A) : A)
    return minimum(real.(res.values))
end

"""
`largest_real_eigval(A, kmaxiter[, noise=1])`

Given a hermitian matrix `A`, attempts to compute the 
most positive (real) eigenvalue. First, uses Krylov iteration
with no shift-invert, and if not successful, calls a full, 
dense, eigensolve.
"""
function largest_real_eigval(A::AbstractMatrix, kmaxiter, noise=1)
    # Do not invert
    @assert ishermitian(A)
    res = eigsolve(x -> A*x, size(A,1), 1, :LM, eltype(A), ishermitian=true, maxiter=kmaxiter, krylovdim=size(A,1))
    if res[3].converged >= 1
        return res[1][1] 
    end
    if noise >= 1
        println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
    end
    # Perform brute eigen
    res = eigen!(issparse(A) ? collect(A) : A)
    return maximum(real.(res.values))
end

"""
`smallest_real_pos_eigpair(A, kmaxiter[, noise=1])`

Given a hermitian, positive definite matrix `A`, attempts to compute the 
smallest (real) eigenvalue and eigenvector. First, uses Krylov iteration
with shift-invert around zero, and if not successful, calls a full, 
dense, eigensolve. Returns a tuple with the first component being the 
eigenvalue, and the second component being the eigenvector.
"""
function smallest_real_pos_eigpair(A::AbstractMatrix, kmaxiter, noise=1)
    # Try invert around 0
    invert = (x -> A \ x) 
    @assert ishermitian(A)
    res = eigsolve(invert, size(A,1), 1, :LM, eltype(A), ishermitian=true, maxiter=kmaxiter, krylovdim=size(A,1))
    if res[3].converged >= 1
        return (1 / res[1][1], res[2][1])
    end
    if noise >= 1
        println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
    end
    # Perform brute eigen
    res = eigen!(issparse(A) ? collect(A) : A, sortby=real)
    return (minimum(real.(res.values)), res.vectors[:,1])
end

"""
`smallest_sval(A, kmaxiter[, noise=1])`

Given a matrix `A`, attempts to compute the smallest singular
value of it by Krylov iteration and inversion around 0. If
unsuccessful, computes a full, dense svd.
"""
function smallest_sval(A::AbstractMatrix, kmaxiter, noise=1)
    invert = (x -> A \ x)
    invert_adj = (x -> A' \ x) 
    res = svdsolve((invert, invert_adj), size(A,1), 1, :LR, eltype(A), maxiter=kmaxiter, krylovdim=size(A,1))
    if res[4].converged >= 1
        return (1 / res[1][1])
    end
    if noise >= 1
        println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
    end
    # Perform brute eigen
    res = svd(issparse(A) ? collect(A) : A)
    return res.S[end]
end

"""
`orthonormalize_mgs2!(u, V)`

Given a matrix `V`, and a new vector `u`, orthogonalize `u` 
with respect to the columns of `V`, and computes its
norm `nu`. If `nu != 0`, divides `u` by `nu` and returns 
`nu`. If `nu == 0`, then `u` lives in the span of `V`, and
`0` is returned.
"""
function orthonormalize_mgs2!(u::AbstractVector,V::AbstractMatrix)
    for v in eachcol(V)
        u .-= dot(v, u) .* v
    end
    for v in eachcol(V)
        u .-= dot(v, u) .* v
    end
    nu = norm(u)
    if nu != 0
        u ./= nu
    end
    return nu
end