using Printf
using Plots 
using LinearAlgebra

struct EIM
    g::Function
    gis::Vector
    xis::Vector{Int}
    errs::Vector{Float64}
    ps::Vector
    makeθgis::Function
    Qg::Int
    deim::Bool
end

function Base.show(io::Core.IO, eim::EIM)
    if eim.deim
        res = "D"
    else
        res = ""
    end
    res *= "EIM approximation for (vectorized) function g(p)\n"
    res *= "with $(eim.Qg) affine terms.\n"
    res *= "Form affine parametrized terms with eim.makeθgis(p),\n"
    res *= "the spatial affine terms are in eim.gis, and form the\n"
    res *= "EIM approximant with eim(p)."
    print(io, res)
end

function (eim::EIM)(p)
    # Return the eim approximation
    γ = eim.makeθgis(p)
    res = γ[1] .* eim.gis[1]
    for i in 2:eim.Qg
        res .+= γ[i] .* eim.gis[i]
    end
    return res
end

function eim(g::Function,
             param_disc::Union{<:AbstractMatrix,<:AbstractVector},
             ϵ=1e-1;
             maxM=100,
             noise=1)
    if param_disc isa AbstractMatrix
        param_disc = eachcol(param_disc)
    end
    gvec = g(param_disc[1])
    S = zeros(length(gvec), length(param_disc))
    S[:,1] .= gvec
    for i in eachindex(param_disc)[2:end]
        p = param_disc[i]
        S[:,i] .= g(p)
    end
    # Loop thru param_disc, choose largest norm g 
    largestg = S[:,1]
    largest_normg = 0
    gvec = similar(largestg)
    p1 = param_disc[1]
    for i in eachindex(param_disc)
        gvec .= S[:,i]
        p = param_disc[i]
        normg = norm(gvec, Inf)
        if normg > largest_normg
            largest_normg = normg
            largestg .= gvec
            p1 = p
        end
    end
    # Find first interpolating x-value, normalize
    _, x1 = findmax(abs.(largestg))
    ρ1 = largestg ./ largestg[x1]
    # Form interpolation matrix and vectors 
    ps = [p1]
    xs = [x1]
    ρs = [ρ1] 
    BM = zeros(maxM, maxM)
    BM[1,1] = 1.0
    gM = zeros(maxM)
    function interp(p,xs,ρs,BM,gM)
        M = length(xs)
        gvec .= g(p)
        for i in 1:M
            gM[i] = gvec[xs[i]] 
        end
        γ = view(BM,1:M,1:M) \ view(gM,1:M)
        res = γ[1] .* ρs[1]
        for j in 2:length(ρs)
            res .+= γ[j] .* ρs[j]
        end
        return res
    end
    # Loop
    j = 2
    errs = [largest_normg]
    while j <= maxM
        # Loop thru param_disc, choose largest norm residual
        largest_res = similar(gvec)
        largest_norm_res = 0.0
        pj = param_disc[1]
        res = gvec
        for (i,p) in enumerate(param_disc)
            if p in ps
                continue
            end
            res .= S[:,i] - interp(p, xs, ρs, BM, gM)
            norm_res = norm(res, Inf)
            if norm_res > largest_norm_res
                largest_norm_res = norm_res
                largest_res .= res
                pj = p
            end
        end
        push!(errs, largest_norm_res)
        if largest_norm_res < ϵ
            break
        end
        # Find first interpolating x-value, normalize
        _, xj = findmax(abs.(largest_res))
        ρj = largest_res ./ largest_res[xj]
        # Append to interpolation matrix and vectors
        push!(ps, pj)
        push!(xs, xj)
        push!(ρs, ρj)
        # BM[1:j-1,1:j-1] already filled out 
        for k in 1:j
            BM[j,k] = ρs[k][xs[j]]
        end
        # BM = zeros(j,j)
        # for k in 1:j
        #     BM[k,k] = 1
        #     for l in k+1:j
        #         BM[l,k] = ρs[k][xs[l]]
        #     end
        # end
        if noise >= 1
            @printf("(%d): maxres=%.2e\n",j,largest_norm_res)
        end
        j += 1
    end
    M = length(xs)
    BM = BM[1:M,1:M]
    makeθgis(p) = begin
        gvec .= g(p)
        return BM \ view(gvec,xs)
    end
    return EIM(g, ρs, xs, errs, ps, makeθgis, M, false)
end

function deim(g::Function,
              param_disc::Union{<:AbstractMatrix,<:AbstractVector},
              ϵ=1e-1;
              maxM=100,
              noise=1)
    if param_disc isa AbstractMatrix
        param_disc = eachcol(param_disc)
    end
    # Loop thru param_disc, form snapshots
    gvec = g(param_disc[1])
    S = zeros(length(gvec), length(param_disc))
    S[:,1] .= gvec
    for i in 2:length(param_disc)
        p = param_disc[i]
        S[:,i] .= g(p)
    end
    # Use POD to form snapshots 
    svdS = svd(S)
    M = min(maxM, size(svdS.U)[1])
    U = svdS.U[:,1:M]
    # Find first interpolating x-value, normalize
    _, x1 = findmax(abs.(view(U,:,1)))
    # Form interpolation matrix and vectors 
    xs = [x1]
    ρs = [view(U,:,1)] 
    BM = ones(1,1)
    BM[1,1] = U[x1,1]
    for j in 2:M
        # Find first interpolating x-value, normalize
        res = view(U,:,j) .- (view(U,:,1:j-1) * (view(U,xs,1:j-1) \ view(U,xs,j)))
        # norm_residual = norm(view(U,:,j))
        # if norm_residual < ϵ
        #     break
        # end
        # @printf("(%d): maxres=%.2f\n",j,norm_residual)
        _, xj = findmax(abs.(res))
        # Append to interpolation matrix and vectors
        push!(xs, xj)
        push!(ρs, view(U,:,j))
        # TODO: Naive expansion of BM
    end
    
    function interp(p,xs)
        M = length(xs)
        gM = zeros(M)
        gvec .= g(p)
        for i in 1:M
            gM[i] = gvec[xs[i]] 
        end
        γ = view(U,xs,1:M) \ gM
        res = γ[1] .* view(U,:,1)
        for j in 2:M
            res .+= γ[j] .* view(U,:,j)
        end
        return res
    end
    M = length(xs)
    BM = U[xs,1:M]
    makeθgis(p) = begin
        gvec .= g(p)
        return BM \ view(gvec,xs)
    end
    errs = Float64[]
    ps = []
    return EIM(g, ρs, xs, errs, ps, makeθgis, M, true)
end

# Example
xs = range(1,5,1000)
# g(p) = p^2 .* sin.(pi .* xs) .+ p .* sqrt.(xs)
g(p) = sin.(p .* xs) .+ sqrt.(xs ./ p)

param_disc = range(1,5,1000)

p1 = plot()
for p in param_disc
    plot!(xs, g(p), label=false, alpha=0.1, c="black")
end
title!("g(p)")

@time eimmethod = eim(g, param_disc, 1e-16, maxM=20, noise=0)
@time deimmethod = deim(g, param_disc, 1e-4, maxM=20)

plot(eimmethod.errs, yaxis=:log)

p2 = plot()
for p in param_disc
    plot!(xs, eimmethod(p), label=false, alpha=0.25)
end
title!("eim(p)")

p4 = plot()
for p in param_disc
    plot!(xs, g(p) .- eimmethod(p), label=false, alpha=0.25)
end
scatter!(xs[eimmethod.xis], zeros(length(eimmethod.xis)), label=false)
title!("g(p) - eim(p)")

p5 = plot()
errs = [norm(g(p) - eimmethod(p)) for p in param_disc]
histogram!(errs)
title!("||g(p) - eim(p)||")

p3 = plot()
for p in param_disc
    plot!(xs, deimmethod(p), label=false, alpha=0.25)
end
title!("deim(p)")

p6 = plot()
for p in param_disc
    plot!(xs, g(p) .- deimmethod(p), label=false, alpha=0.25)
end
scatter!(xs[deimmethod.xis], zeros(length(deimmethod.xis)), label=false)
title!("g(p) - deim(p)")

p7 = plot()
errs = [norm(g(p) - deimmethod(p)) for p in param_disc]
histogram!(errs)
title!("||g(p) - deim(p)||")
