using LinearAlgebra
using Printf
include("successive_constraint.jl")
include("residual_norm.jl")

"""
`Greedy_RB_Affine_Linear`

Struct for containing initialized greedy reduced-basis method
for parametrized problem A(p) x = b(p) with affine parameter
dependence
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.

Uses Galerkin projection onto span of columns of `V`, 
`V' A(p) V x_r = V' b(p)`, with `V x_r ≈ x = A(p)^(-1) b(p)`.

Given a new parameter vector, `p`, and an object
`greedy_sol::Greedy_RB_Affine_Linear`, form the reduced
basis solution with `greedy_sol(p[, full=true])`. 
"""
struct Greedy_RB_Affine_Linear
    scm_init::SCM_Init
    res_init::Affine_Residual_Init
    Ais::AbstractVector
    makeθAi::Function
    bis::AbstractVector
    makeθbi::Function
    param_disc::AbstractVector
    params_greedy::AbstractVector
    V::AbstractMatrix
    VtAVis::AbstractVector{<:AbstractMatrix}
    Vtbis::AbstractVector{<:AbstractVector}
    ϵ::Real
end

function Base.show(io::Core.IO, greedy_sol::Greedy_RB_Affine_Linear)
    res = "Initialized reduced basis method for parametrized problem A(p) x = b(p) with affine parameter dependence:\n"
    res *= "    A(p) = ∑ makeθAi(p,i) Ais[i],\n"
    res *= "    b(p) = ∑ makeθbi(p,i) bis[i].\n"
    res *= "Galerkin projection is used onto an $(size(greedy_sol.V)[2]) dimensional space:\n"
    res *= "    V' A(p) V x_r = V' b(p),\n"
    res *= "    V x_r ≈ x = A(p)^(-1) b(p)."
    print(io, res)
end

"""
`(greedy_sol::Greedy_RB_Affine_Linear)(p[, full=true])`

Forms the reduced basis solution for the parameter vector `p`,
`(V' A(p) V)^(-1) V' b(p)`. If full is set to false, then returns
the shortened vector `(V' A(p))^(-1) V' b(p)`.
"""
function (greedy_sol::Greedy_RB_Affine_Linear)(p, full=true)
    VtAV = zeros(size(greedy_sol.VtAVis[1]))
    for i in eachindex(greedy_sol.VtAVis)
        VtAV .+= greedy_sol.makeθAi(p,i) .* greedy_sol.VtAVis[i]
    end
    Vtb = zeros(length(greedy_sol.Vtbis[1]))
    for i in eachindex(greedy_sol.Vtbis)
        Vtb .+= greedy_sol.makeθbi(p,i) .* greedy_sol.Vtbis[i]
    end
    x_r =  VtAV \ Vtb
    if full
        return greedy_sol.V * x_r
    else
        return x_r
    end
end

"""
`GreedyRBAffineLinear(scm_init, Ais, makeθAi, bis, makeθbi[, ϵ=1.0, param_disc=nothing; max_snapshots=-1, noise=1])`

Constructs a `Greedy_RB_Affine_Linear` object for parametrized problem A(p) x = b(p) with affine parameter
dependence:
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.
Uses `scm_init` to approximate the stability factor for each parameter in the discretization, chooses first 
parameter to be one with smallest stability factor. Afterwards, utilizes the affine decomposition of `A(p)` 
and `b(p)` to approximate the norm of the residual, and compute upper-bounds for the error:
`||x - V x_r|| <= ||b(p) - A(p) V x_r|| / σ_min(A(p))`.

Greedily chooses parameters and then forms truth solution for parameter that results in largest upper-bound
for error. Then computes `||x - V x_r||`, and loops, adding to reduced basis, `V`, until the truth error
is less than `ϵ`.

If `param_disc` not passed in, then params will be taken from `scm_init`. If `max_snapshots` specified, will
halt after that many if `ϵ` accuracy not yet reached. `noise` determines amount of printed output, `0` for no
output, `1` for basic, and `2` for more.
"""
function GreedyRBAffineLinear(scm_init::SCM_Init,
                              Ais::AbstractVector,
                              makeθAi::Function,
                              bis::AbstractVector,
                              makeθbi::Function,
                              ϵ=1e-2,
                              param_disc::Union{Matrix,Vector,Nothing}=nothing;
                              max_snapshots=-1,
                              noise=1)
    if isnothing(param_disc)
        params = scm_init.tree.data
    elseif param_disc isa Matrix
        params = [param_disc[:,i] for i in 1:size(param_disc)[2]]
    else
        params = param_disc
    end
    # Choose a parameter vector p to begin with
    truth_sol(p) = begin
        A = zeros(size(Ais[1]))
        for i in eachindex(Ais)
            A .+= makeθAi(p,i) .* Ais[i]
        end
        b = zeros(length(bis[1]))
        for i in eachindex(bis)
            b .+= makeθbi(p,i) .* bis[i]
        end
        return A \ b
    end
    # Begin iteration by parameter with minimum stability factor
    if noise >= 1
        @printf("Beginning greedy selection, looping until truth error less than ϵ=%.2e\n----------\n",ϵ)
    end
    maxerr = 0
    p1 = nothing
    for p in params
        stability_factor = find_sigma_bounds(scm_init, p, ϵ*10)[1]
        err = 1.0 / stability_factor
        if err > maxerr
            p1 = p
            maxerr = err
        end
    end
    x = truth_sol(p1)
    x = x ./ norm(x)
    V0 = reshape(x, (length(x), 1))
    VtAVis = [V0' * Ai * V0 for Ai in Ais]
    Vtbis = [V0' * b for b in bis]
    approx_sol(p,VtAVis,Vtbis) = begin
        VtAV = zeros(size(VtAVis[1]))
        for i in eachindex(VtAVis)
            VtAV .+= makeθAi(p,i) .* VtAVis[i]
        end
        Vtb = zeros(length(Vtbis[1]))
        for i in eachindex(Vtbis)
            Vtb .+= makeθbi(p,i) .* Vtbis[i]
        end
        return VtAV \ Vtb
    end
    # Initialize res_init to compute residual norm
    res_init = residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, V0)
    if noise >= 1
        print("k=1, first parameter value chosen, initialized for greedy search\n")
    end
    # Greedily loop over every parameter value
    ps = [p1]
    if max_snapshots == -1
        max_snapshots = length(params)
    end
    for k in 2:max_snapshots
        maxerr = 0
        maxp = nothing
        for p in params
            if p in ps
                continue
            end
            stability_factor = find_sigma_bounds(scm_init, p, ϵ*10)[1]
            x_r = approx_sol(p,VtAVis,Vtbis)
            res_norm = residual_norm_affine_online(res_init, x_r, p)
            err = res_norm / stability_factor
            if err > maxerr
                maxp = p
                maxerr = err
            end
        end
        # Compute full solution to add to V
        x = truth_sol(maxp)
        x_r = approx_sol(maxp,VtAVis,Vtbis)
        trueerr = norm(x .- V0*x_r)
        # Make orthogonal to v1,...,vn
        for i in 1:size(V0)[2]
            x .-= (V0[:,i]'x) .* V0[:,i]
        end
        nx = norm(x)
        x = x ./ nx
        for i in eachindex(Ais)
            row = x' * Ais[i] * V0
            col = zeros(length(row) + 1)
            col[1:end-1] .= V0' * Ais[i] * x
            col[end] = x'  * Ais[i] * x
            VtAVis[i] = hcat(vcat(VtAVis[i], row), col)
        end
        for i in eachindex(bis)
            Vtbis_end = x' * bis[i]
            push!(Vtbis[i], Vtbis_end)
        end
        add_col_to_V(res_init, x)
        V0 = hcat(V0, x)
        push!(ps, maxp)
        if noise >= 1
            @printf("k=%d, truth error = %.4e, upperbound error = %.4e\n",k,trueerr,maxerr)
        end
        # Break if truth error less than ϵ
        if trueerr < ϵ
            break
        end
    end
    if noise >= 1
        @printf("----------\nCompleted greedy selection after k=%d iterations\n",length(ps))
    end
    return Greedy_RB_Affine_Linear(scm_init, res_init, Ais, makeθAi, bis,
                                   makeθbi, params, ps, V0, VtAVis,
                                   Vtbis, ϵ)
end