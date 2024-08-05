"""
`AffineParametrizedArray(arrays, makeθi[, precompθ=false])`

Struct for containing an affine parametrized array `A(p)` of the form
`A(p) = ∑ makeθi(p,i) * arrays[i]`.

Arrays are stored in the vector `arrays` for quick recomputation for new parameter
values. Each must be of the same size and dimension so that they can be 
broadcast summed.

If precompθ set to `false` (default):

The function `makeθi(p,i)` takes in a parameter object first,
and second an index `i=1,...,length(arrays)`, and returns a scalar

If `precompθ` set to `true`:

The function `makeθi(p)` takes in a parameter object, and returns
a vector of each of the affine terms.

Given `aparr <: AffineParametrizedArray`, and a new parameter value `p`, 
can form the full array `A(p)` by calling `aparr(p)`.
"""
struct AffineParametrizedArray
    arrays::AbstractVector
    makeθi::Function
    precompθ::Bool
    AffineParametrizedArray(arrays::AbstractVector,
                            makeθi::Function,
                            precompθ::Bool=false) = new(arrays,makeθi,precompθ)
end

APArray = AffineParametrizedArray

@doc (@doc AffineParametrizedArray) APArray

function Base.show(io::Core.IO, aparr::APArray)
    res = "$(size(aparr.arrays[1])) affine parameter dependent array with $(length(aparr.arrays)) terms"
    print(io, res)
end

"""
`formArray!(aparr, arr, p)`

Given an array `arr` with the same dimensions as the arrays in the APArray `aparr`,
form `A(p)` and place its values in `arr`.
"""
function formArray!(aparr::AffineParametrizedArray,arr,p)
    if aparr.precompθ
        θis = aparr.makeθi(p)
        arr .= θis[1] * aparr.arrays[1]
        for i in eachindex(aparr.arrays)[2:end]
            arr .+= θis[i] * aparr.arrays[i]
        end
    else
        arr .= aparr.makeθi(p,1) * aparr.arrays[1]
        for i in eachindex(aparr.arrays)[2:end]
            arr .+= aparr.makeθi(p,i) * aparr.arrays[i]
        end
    end
end

function (aparr::AffineParametrizedArray)(p)
    arr = similar(aparr.arrays[1])
    formArray!(aparr, arr, p)
    return arr
end

"""
`eim(arrFun, param_disc[, ϵ=1e-2; maxM=100, noise=1])`

Method for constructing an `APArray` object from a non-affinely parameter
dependent matrix `arrFun(p)` by empirical interpolation. 

`param_disc` must be a matrix with columns as parameter vectors, or a vector
with elements as parameters.

Loops over the given parameter discretization until a maximum ∞-norm error 
of `ϵ` is achieved over the entire discretization, or until the maximum number
of parameter values are chosen, given by `maxM`.

`noise` dictates the amount of printed output. Set `0` for no output,
`1` for some output, `≥2` for most.
"""
function eim(arrFun::Function,
             param_disc::Union{<:AbstractMatrix,<:AbstractVector},
             ϵ=1e-2;
             maxM=100,
             noise=1)
    if param_disc isa AbstractMatrix
        param_disc = eachcol(param_disc)
    end
    if noise >= 2
        println("Beginning computation of all array values")
    end
    S = [arrFun(param_disc[1])]
    maxarrnorm = -1.0
    maxarridx = 0
    for i in eachindex(param_disc)[2:end]
        arr = arrFun(param_disc[i])
        push!(S, arr)
        arrnorm = norm(arr, Inf)
        if arrnorm > maxarrnorm
            maxarrnorm = arrnorm
            maxarridx = i
        end
    end
    if noise >= 2
        println("Completed computation of all array values")
    end
    # Find first interpolating x-value, normalize
    _, x1 = findmax(abs.(S[maxarridx]))
    ρ1 = S[maxarridx] ./ S[maxarridx][x1]
    # Form interpolation matrix and vectors 
    ps = [param_disc[maxarridx]]
    xs = [x1]
    ρs = [ρ1] 
    BM = zeros(maxM, maxM)
    BM[1,1] = 1.0
    gM = zeros(maxM)
    function interp(pidx,xs,ρs,BM,gM)
        M = length(xs)
        for i in 1:M
            gM[i] = S[pidx][xs[i]] 
        end
        γ = view(BM,1:M,1:M) \ view(gM,1:M)
        res = γ[1] .* ρs[1]
        for j in eachindex(ρs)[2:end]
            res .+= γ[j] .* ρs[j]
        end
        return res
    end
    # Loop
    j = 2
    errs = [maxarrnorm]
    arrtmp = similar(S[1])
    maxarr = similar(arrtmp)
    while j <= maxM
        # Loop thru param_disc, choose largest norm residual
        # largest_res = similar(arr)
        maxarrnorm = -1.0
        maxarridx = 0
        for (i,p) in enumerate(param_disc)
            if p in ps
                continue
            end
            arrtmp .= S[i] .- interp(i, xs, ρs, BM, gM)
            arrnorm = norm(arrtmp, Inf)
            if arrnorm > maxarrnorm
                maxarrnorm = arrnorm
                maxarridx = i
                maxarr .= arrtmp
            end
        end
        push!(errs, maxarrnorm)
        if noise >= 1
            @printf("(%d): maxres=%.2e\n",j,maxarrnorm)
        end
        if maxarrnorm < ϵ
            break
        end
        # Find first interpolating x-value, normalize
        _, xj = findmax(abs.(maxarr))
        ρj = maxarr ./ maxarr[xj]
        # Append to interpolation matrix and vectors
        push!(ps, param_disc[maxarridx])
        push!(xs, xj)
        push!(ρs, ρj)
        for k in 1:j
            BM[j,k] = ρs[k][xs[j]]
        end
        j += 1
    end
    M = length(xs)
    BM = BM[1:M,1:M]
    gM = zeros(M)
    makeθgi(p) = begin
        arrtmp .= arrFun(p)
        for i in 1:M
            gM[i] = arrtmp[xs[i]] 
        end
        return BM \ view(arrtmp,xs)
    end
    return AffineParametrizedArray(ρs, makeθgi, true)
end