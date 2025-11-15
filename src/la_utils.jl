struct FullLU
    P::AbstractVector
    Q::AbstractVector
    L::AbstractMatrix
    U::AbstractMatrix
end

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
    return FullLU(P,L,U,Q)
end

"""
`reig(A::AbstractMatrix, [B=I; which=:L, kmaxiter=1000, noise=0, krylovsteps=8, eps=1e-14, reldifftol=0.9])`

Given (symmetric) matrices `A` and `B`, computes a real eigenvalue

``A x = λ B x``.

- `which=:L` corresponds to the largest eigenvalue
- `which=:S` corresponds to the smallest eigenvalue
- `which=:SP` corresponds to the smallest positive eigenvalue

Attempts to do this by shift-and-invert using `Arpack.jl` where the shifts are
determined by the eigenvalue seeked and the Gershgorin disks of `A`.

Parameter `eps` is used to perturb shift parameters by relative amount to attempt
to ensure no singular shifts. If the relative gap between the shift and the found
eigenvalue is greater than `reldifftol`, attempts to resolve shifting about the 
found eigenvalue.
"""
function reig(A::AbstractMatrix, B::Union{AbstractMatrix,UniformScaling}=I; which=:L, kmaxiter=1000, noise=0, krylovsteps=8, eps=1e-14, reldifftol=0.9)
    # Compute Gershgorin disks
    mingd = Inf; maxgd = -Inf
    mingd_center = 0.0; maxgd_center = 0.0
    for i in 1:size(A)[1]
        newmingd = real(A[i,i]) - (sum(abs.(view(A,i,:))) - abs(A[i,i]))
        if newmingd < mingd
            mingd = newmingd
            mingd_center = real(A[i,i])
        end
        newmaxgd = real(A[i,i]) + (sum(abs.(view(A,i,:))) - abs(A[i,i]))
        if newmaxgd > maxgd
            maxgd = newmaxgd
            maxgd_center = real(A[i,i])
        end
    end
    log_tilde(x) = x >= 1 ? log(x) : (x <= -1 ? -log(-x) : 0)
    exp_tilde(x) = x > 0 ? exp(x) : (x < 0 ? -exp(-x) : 0)
    sigmas = Float64[]
    if which == :S
        if mingd < 0 && 0 < mingd_center
            append!(sigmas, exp_tilde.(range(log_tilde(mingd - 1.5), 0.0, div(krylovsteps,2))))
            append!(sigmas, exp_tilde.(range(0.0, log_tilde(mingd_center), div(krylovsteps,2))))
        else
            append!(sigmas, exp_tilde.(range(log_tilde(mingd - 1.5), log_tilde(mingd_center), krylovsteps)))
        end
    elseif which == :L
        if maxgd > 0 && 0 > maxgd_center
            append!(sigmas, exp_tilde.(range(log_tilde(maxgd + 1.5), 0.0, div(krylovsteps,2))))
            append!(sigmas, exp_tilde.(range(0.0, log_tilde(maxgd_center), div(krylovsteps,2))))
        else
            append!(sigmas, exp_tilde.(range(log_tilde(maxgd + 1.5), log_tilde(maxgd_center), krylovsteps)))
        end
    elseif which == :SP
        push!(sigmas, max(abs(eps), mingd - abs(mingd*eps)))
    else
        error("Unknown which=$which, choose between (:S, :L, :SP) for smallest, largest, or smallest positive (real) eigenvectors")
    end
    unique!(sigmas)
    for (i,sigma) in enumerate(sigmas)
        try
            res = eigs(A, B, which=:LM, sigma=sigma, ritzvec=true, nev=1, maxiter=kmaxiter)
            eg = real(res[1][1]); egval = res[2][:,1]
            close = i == length(sigmas) || (i < length(sigmas) && eg < sigmas[i+1])
            if !close
                continue
            end
            if abs((sigma - eg) / max(abs(eg),abs(eps))) > reldifftol
                sigma = eg - abs(eg*eps)
                # Try again to recenter about eg
                res = eigs(A, B, which=:LM, sigma=sigma, ritzvec=true, nev=1, maxiter=kmaxiter)
                eg = real(res[1][1]); egval = res[2][:,1]
                return eg, egval
            else
                return eg, egval
            end
        catch e
            if isa(e, SingularException) || isa(e, ZeroPivotException) || isa(e,XYAUPD_Exception)
                # Try at new value
                continue
            else
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
    res = eigen(issparse(A) ? collect(A) : A, isa(B, UniformScaling) ? B(size(A,1)) : issparse(B) ? collect(B) : B)
    if which == :S
        λ,i = findmin(x -> real(x), res.values)
        return real(λ), view(res.vectors, :, i)
    elseif which == :L
        λ,i = findmax(x -> real(x), res.values)
        return real(λ), view(res.vectors, :, i)
    else # which == :SP
        λ,i = findmin(x -> x < 0 ? Inf : real(x), res.values)
        return real(λ), view(res.vectors, :, i)
    end
end

"""
`smallest_sval(A[; kmaxiter=1000, noise=0])`

Given a matrix `A`, attempts to compute the smallest singular
value of it by Krylov iteration and inversion around 0. If
unsuccessful, computes a full, dense svd.
"""
function smallest_sval(A::AbstractMatrix; kmaxiter=1000, noise=0)
    return reig(A'A, which=:SP, kmaxiter=kmaxiter, noise=noise)[1]
end

"""
`largest_sval(A[; kmaxiter=1000, noise=0])`

Given a matrix `A`, attempts to compute the largest singular
value of it by Krylov iteration. If unsuccessful, computes a 
full, dense svd.
"""
function largest_sval(A::AbstractMatrix; kmaxiter=1000, noise=0)
    try
        return svds(A, maxiter=kmaxiter, nsv=1)[1].S[1]
    catch _
        if noise >= 1
            println("svds did not converge, computing full SVD")
        end
        return maximum(svd(issparse(A) ? collect(A) : A).S)
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

function max_real_numerical_range(A)
    return largest_real_eigval(0.5 .* (A .+ A'), 1000)
end

function nrange(B; nk=1, thmax=32)
        function rq(A, x)
                return x'*A*x/(x'*x)
        end
        thmax -= 1 # the function uses thmax + 1 angles

        (n, p) = size(B)
        if n != p
                DimensionMismatch("Matrix must be square.")
        end

        z = Vector{ComplexF64}(undef, 2*thmax + 1)
        F = eigen(collect(B))
        e = F.values
        f = ComplexF64[]

        # filter out cases where B is Hermitian or skew-Hermitian
        if B == B'
                f = [minimum(e), maximum(e)]
        elseif B == -B'
                e = imag(e)
                f = [minimum(e), maximum(e)]
                e *= im
                f *= im
        else
                for m = 1:nk
                        ns = n + 1 - m
                        A = B[1:ns, 1:ns]
                        for i = 0:thmax
                                th = i/thmax*pi
                                Ath = exp(im*th)*A
                                H = 0.5*(Ath + Ath')
                                X = eigen(collect(H), sortby = x -> real(x))
                                V = X.vectors
                                z[i + 1] = rq(A, V[:, 1])
                                z[i + 1 + thmax] = rq(A, V[:, ns])
                        end
                        f = [f; z]
                end
                # join up the boundary
                f = [f; f[1]]
        end
        if thmax == 0
                f = e
        end

        return f, e
end