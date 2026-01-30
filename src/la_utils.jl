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

function compute_shifts(A::AbstractMatrix, B::Union{AbstractMatrix,UniformScaling}=I; which=:L, randdisks=1000, ignoreB=false, minstep=1.0, krylovsteps=8, eps=1e-14)
    N = size(A,1)
    colidxs = (randdisks > N) ? (1:N) : (1:N)[randperm(N)[1:randdisks]]
    # Compute Gershgorin disks
    mingd = Inf; maxgd = -Inf
    mingd_center = 0.0; maxgd_center = 0.0
    Bfact = begin
        if ignoreB
            I
        elseif isa(B, UniformScaling)
            B
        else
            factorize(B)
        end
    end
    for i in colidxs
        col = Bfact \ collect(view(A, :, i))
        gd_center = real(col[i])
        newmingd = 2 * gd_center - sum(abs.(col))
        if newmingd < mingd
            mingd = newmingd
            mingd_center = gd_center
        end
        newmaxgd = sum(abs.(col))
        if newmaxgd > maxgd
            maxgd = newmaxgd
            maxgd_center = gd_center
        end
    end
    # To avoid zero pivot error 
    mingd *= (mingd < 0 ? (1 + eps) : (1 - eps))
    mingd_center *= (mingd_center < 0 ? (1 + eps) : (1 - eps))
    if -1 <= mingd && mingd <= 1
        mingd = -1 - eps
    end
    maxgd *= (maxgd < 0 ? (1 - eps) : (1 + eps))
    maxgd_center *= (maxgd_center < 0 ? (1 - eps) : (1 + eps))
    if -1 <= maxgd && maxgd <= 1
        maxgd = 1 + eps
    end
    log_tilde(x) = x >= 1 ? log(x) : (x <= -1 ? -log(-x) : 0)
    exp_tilde(x) = x > 0 ? exp(x) : (x < 0 ? -exp(-x) : 0)
    sigmas = Float64[]
    if which == :S
        if mingd < 0 && 0 < mingd_center
            append!(sigmas, exp_tilde.(range(log_tilde(mingd), 0.0, div(krylovsteps,2))))
            append!(sigmas, exp_tilde.(range(0.0, log_tilde(mingd_center), div(krylovsteps,2))))
        else
            append!(sigmas, exp_tilde.(range(log_tilde(mingd), log_tilde(mingd_center), krylovsteps)))
        end
    elseif which == :L
        if maxgd > 0 && 0 > maxgd_center
            append!(sigmas, exp_tilde.(range(log_tilde(maxgd), 0.0, div(krylovsteps,2))))
            append!(sigmas, exp_tilde.(range(0.0, log_tilde(maxgd_center), div(krylovsteps,2))))
        else
            append!(sigmas, exp_tilde.(range(log_tilde(maxgd), log_tilde(maxgd_center), krylovsteps)))
        end
    elseif which == :SP
        append!(sigmas, exp_tilde.(range(0.0, log_tilde(mingd_center > 0 ? mingd_center : max(0.0, maxgd_center)), div(krylovsteps,2))) .- abs(eps))
    else
        error("Unknown which=$which, choose between (:S, :L, :SP) for smallest, largest, or smallest positive (real) eigenvectors")
    end
    unique!(sigmas)
    sigmas_final = [sigmas[1]]
    for i in 2:length(sigmas)
        if abs(sigmas[i] - sigmas_final[end]) >= minstep
            push!(sigmas_final, sigmas[i])
        end
    end
    return sigmas_final
end

# Assumes the problem is hermitian/symmetric, i.e. eigenvalues are real
function shift_invert_attempt(A::AbstractMatrix, B::Union{AbstractMatrix,UniformScaling}=I, egwith=:arnoldimethod; which=:L, sigma=nothing, kmaxiter=1000, restarts=100, noise=0)
    if isnothing(sigma) && which == :SP
        return 0.0, zeros(1), false
    end
    if egwith == :arnoldimethod
        # Seems that `\` for Cholesky factorization not implemented until 1.12
        factorize_method = begin
            if VERSION.major <= 1 && VERSION.minor < 12
                lu
            else
                factorize
            end
        end
        if isnothing(sigma)
            whichhere = which == :L ? :LR : :SR
            Binvmap = isa(B, UniformScaling) ? LinearMap(inv(B), size(A,1)) : InverseMap(factorize_method(B))
            F = Binvmap * LinearMap(A)
            res = partialschur(F, which=whichhere, nev=1, restarts=restarts)
            if res[2].converged
                egval = real(res[1].R[1])
                egvec = view(res[1].Q,:,1)
                return egval, egvec, true
            else
                if noise >= 2
                    println("Direct eigenvalue iteration did not converge")
                end
                return 0.0, zeros(0), false
            end
        else
            try
                F = InverseMap(factorize_method(A - sigma * B)) * (isa(B, UniformScaling) ? LinearMap(B, size(A,1)) : LinearMap(B))
                res = partialschur(F, which=:LM, nev=1, restarts=restarts)
                if res[2].converged
                    egval = 1 / real(res[1].R[1]) + sigma
                    egvec = view(res[1].Q,:,1)
                    return egval, egvec, true
                else
                    if noise >= 2
                        @printf("Eigenvalue iteration did not converge about sigma=%.2e\n", sigma)
                    end
                    return 0.0, zeros(0), false
                end
            catch e
                if noise >= 2
                    println("Eigenvalue iteration errored with error $e")
                end
                return 0.0, zeros(0), false
            end
        end
    elseif egwith == :arpack
        whichhere = isnothing(sigma) ? (which == :L ? :LR : :SR) : :LM
        try
            res = eigs(A, B, sigma=sigma, which=whichhere, ritzvec=true, nev=1, maxiter=kmaxiter)
            egval = real(res[1][1])
            egvec = res[2][:,1]
            return egval, egvec, true
        catch
            if noise >= 2
                println("Eigenvalue iteration errored with error $e")
            end
            return 0.0, zeros(0), false
        end
    else
        error("Unknown argument egwith=$egwith; must be :arnoldimethod or :arpack")
    end
end

"""
`reig(A::AbstractMatrix, [B=I; which=:L, kmaxiter=1000, noise=0, egwith=:arpack, restarts=100, 
krylovsteps=8, eps=1e-14, randdisks=1000, ignoreB=false, minstep=1.0, force_sigma=nothing])`

Given (symmetric) matrices `A` and `B`, computes a real eigenvalue

``A x = λ B x``.

- `which=:L` corresponds to the largest eigenvalue
- `which=:S` corresponds to the smallest eigenvalue
- `which=:SP` corresponds to the smallest positive eigenvalue

Returns a tuple (`λ`,`v`) where `λ<:Real` is the eigenvalue and `v` the eigenvector.

Attempts to do this by shift-and-invert. Uses `Arpack.jl` if `egwith=:arpack` or
`ArnoldiMethod.jl` if `egwith=:arnoldimethod` where the shifts are
determined by the eigenvalue seeked and the Gershgorin disks of `B⁻¹A`.

If `egwith=:arpack`, `kmaxiter` determines maximum number of iterations. If 
`egwith=:arnoldimethod`, uses `restarts` to determine number of restarts.

NOTE: A randomized algorithm is used by default for determining Gershgorin disks. This
is to speed up the case when `size(A,1) ≫ 1`.

Following paramers relate to selection of shifts:

- `krylovsteps` determines number of logarithmically spaced shifts are attempted
- `ignoreB` determines whether to approximate Gershgorin disks of `B⁻¹A` or `A`
- `randdisks` determines number of (random) columns of `B⁻¹A` or `A` to subselect
to compute Gershgorin disks of, others are ignored. Set `randdisks ≥ size(A,1)` 
for no randomization.
- `minstep` determines the minimum step size between logarithmically spaced shifts
- `force_sigma` forces trying shift and invert about this value if not `nothing`

Parameter `eps` is used to perturb shift parameters by relative amount to attempt
to ensure no singular shifts. If the relative gap between the shift and the found
eigenvalue is greater than `reldifftol` and the absolute gap is greater than `absdifftol`,
attempts to re-solve by shifting about the found eigenvalue.
"""
function reig(A::AbstractMatrix, B::Union{AbstractMatrix,UniformScaling}=I; which=:L, kmaxiter=1000, noise=0, egwith=:arpack, restarts=100,
              krylovsteps=8, eps=1e-14, randdisks=1000, ignoreB=false, minstep=1.0, force_sigma=nothing)
    if iszero(A)
        return 0.0, ones(size(A,1))
    elseif A == B
        return 1.0, ones(size(A,1))
    end
    if which == :L || which == :S
        eg, egval, succeeded = shift_invert_attempt(A, B, egwith, which=which, sigma=nothing, kmaxiter=kmaxiter, restarts=restarts, noise=noise)
        if succeeded
            return eg, egval
        end
    end
    sigmas = compute_shifts(A, B, which=which, krylovsteps=krylovsteps, randdisks=randdisks, 
                            ignoreB=ignoreB, minstep=minstep, eps=eps)
    if isa(force_sigma, Real)
        push!(sigmas, force_sigma)
        unique!(sigmas)
        sort!(sigmas, rev = (which==:L))
    end
    for (i,sigma) in enumerate(sigmas)
        eg, egval, succeeded = shift_invert_attempt(A, B, egwith, which=which, sigma=sigma, kmaxiter=kmaxiter, restarts=restarts, noise=noise)
        if !succeeded
            continue
        end
        # Close if eg is not beyond next shift value
        close = i == length(sigmas)
        if i < length(sigmas) && sigmas[i] < sigmas[i+1]
            close |= eg < sigmas[i+1]
        elseif i < length(sigmas) && sigmas[i] > sigmas[i+1]
            close |= eg > sigmas[i+1]
        end
        if !close
            continue
        end
        if which == :SP && eg < 0
            continue
        end
        return eg, egval
    end
    if noise >= 1
        println("Warning: Eigenvalue iteration did not converge, computing full eigen, may be recommended to update reigkwargs")
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
`smallest_sval(A[; kmaxiter=1000, noise=0, reigkwargs...])`

Given a matrix `A`, attempts to compute the smallest singular
value of it by Krylov iteration and inversion. If
unsuccessful, computes a full, dense svd. See docs for 
`ModelOrderReductionToolkit.reig` for `reigkwargs` options.
Returns the smallest singular value, `σ_min<:Real`
"""
function smallest_sval(A::AbstractMatrix; reigkwargs...)
    return sqrt(reig(A'A, which=:SP; reigkwargs...)[1])
end

"""
`largest_sval(A[; kmaxiter=1000, noise=0])`

Given a matrix `A`, attempts to compute the largest singular
value of it by Krylov iteration. If unsuccessful, computes a 
full, dense svd. Returns the largest singular value, 
`σ_max<:Real`.
"""
function largest_sval(A::AbstractMatrix; reigkwargs...)
    return sqrt(reig(A'A, which=:L; reigkwargs...)[1])
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

function max_real_numerical_range(A; reigkwargs...)
    return reig(0.5 .* (A .+ A'),  which=:L; reigkwargs...)
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