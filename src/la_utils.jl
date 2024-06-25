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
        println(newmingd)
        if newmingd < mingd
            mingd = newmingd
            mingd_center = real(A[i,i])
        end
    end

    log_tilde(x) = x >= 1 ? log(x) : (x <= -1 ? -log(-x) : 0)
    exp_tilde(x) = x > 0 ? exp(x) : -exp(-x)
    exprange = unique(exp_tilde.(range(log_tilde(mingd), log_tilde(mingd_center), krylovsteps)))
    for sigma in exprange
        try
            res = eigs(A, which=:LM, sigma=sigma, nev=1, ritzvec=false, maxiter=kmaxiter)
            return real(res[1][1])
        catch e
            if !isa(e,Arpack.XYAUPD_Exception)
                # Did not converge
                error(e)
            end
        end
        if noise >= 2
            @printf("Krylov iteration did not converge with shift-invert σ=%.2e, reducing\n",mingd)
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
    try
        res = eigs(A, which=:LR, nev=1, ritzvec=false, maxiter=kmaxiter)
        return real(res[1][1])
    catch e
        if !isa(e,Arpack.XYAUPD_Exception)
            error(e)
        end
        if noise >= 1
            println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
        end
        # Perform brute eigen
        res = eigen!(issparse(A) ? collect(A) : A)
        return maximum(real.(res.values))
    end
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
    try
        res = eigs(A, which=:LM, sigma=0, nev=1, ritzvec=true, maxiter=kmaxiter)
        return (real(res[1][1]), view(res[2],:,1))
    catch e
        if !isa(e,Arpack.XYAUPD_Exception)
            error(e)
        end
        if noise >= 1
            println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
        end
        # Perform brute eigen
        res = eigen!(issparse(A) ? collect(A) : A, sortby=real)
        return (minimum(real.(res.values)), res.vectors[:,1])
    end
end

"""
`smallest_sval(A, kmaxiter[, noise=1])`

Given a matrix `A`, attempts to compute the smallest singular
value of it by Krylov iteration and inversion around 0. If
unsuccessful, computes a full, dense svd.
"""
function smallest_sval(A::AbstractMatrix, kmaxiter, noise=1)
    AtA = A'A
    # Try invert around 0
    try
        res = eigs(AtA, which=:LM, sigma=0, nev=1, ritzvec=false, maxiter=kmaxiter)
        return real(res[1][1])
    catch e
        if !isa(e,Arpack.XYAUPD_Exception)
            error(e)
        end
        if noise >= 1
            println("Warning: Krylov iteration did not converge, computing full eigen, may be recommended to increase kmaxiter (currently $(kmaxiter))")
        end
        # Perform brute eigen
        res = eigen!(issparse(AtA) ? collect(AtA) : AtA, sortby=real)
        return minimum(real.(res.values))
    end
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