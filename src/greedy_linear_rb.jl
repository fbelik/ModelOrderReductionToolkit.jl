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
    approx_stability_factor::Function
    res_init::Affine_Residual_Init
    Ais::AbstractVector
    makeθAi::Function
    bis::AbstractVector
    makeθbi::Function
    param_disc::AbstractVector
    params_greedy::AbstractVector
    V::Vector{Vector{Float64}}
    VtAVis::Vector{Vector{Vector{Float64}}}
    Vtbis::Vector{Vector{Float64}}
    ϵ::Real
    VtAV::Matrix{Float64} # Preallocated
    Vtb::Vector{Float64}
end

function Base.show(io::Core.IO, greedy_sol::Greedy_RB_Affine_Linear)
    res = "Initialized reduced basis method for parametrized problem A(p) x = b(p) with affine parameter dependence:\n"
    res *= "    A(p) = ∑ makeθAi(p,i) Ais[i],\n"
    res *= "    b(p) = ∑ makeθbi(p,i) bis[i].\n"
    res *= "Galerkin projection is used onto an $(length(greedy_sol.V)) dimensional space:\n"
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
    greedy_sol.VtAV .= 0.0
    for i in eachindex(greedy_sol.VtAVis)
        θAi = greedy_sol.makeθAi(p,i)
        for j in eachindex(greedy_sol.VtAVis[1])
            greedy_sol.VtAV[:,j] .+= θAi .* greedy_sol.VtAVis[i][j]
        end
    end
    greedy_sol.Vtb .= 0.0
    for i in eachindex(greedy_sol.Vtbis)
        greedy_sol.Vtb .+= greedy_sol.makeθbi(p,i) .* greedy_sol.Vtbis[i]
    end
    x_r = greedy_sol.VtAV \ greedy_sol.Vtb
    x_approx = zeros(length(greedy_sol.V[1]))
    if full
        for i in eachindex(x_r)
            x_approx .+= x_r[i] * greedy_sol.V[i]
        end
        return x_approx
    end
    return x_r
end

"""
`init_affine_rbm(x, Ais, bis, makeθAi, makeθbi)`

Given a vector `x`, generate the matrices `V`, `VtAVi`, and `VtAVis`,
and the vectors `Vtb` and `Vtbis` for preallocation and quick computation
of reduced basis solutions for the problem `A(p)x(p)=b(p)` with affinely
dependent matrix `A(p) = ∑ makeθAi(p,i) Ais[i]` and vector
`b(p) = ∑ makeθbi(p,i) bis[i]`.
"""
function init_affine_rbm(x::AbstractVector, Ais::AbstractVector, bis::AbstractVector, makeθAi::Function, makeθbi::Function)
    V = Vector{Float64}[x]
    VtAVis = Vector{Vector{Float64}}[]
    for Ai in Ais
        VtAVi = Vector{Float64}[Float64[x' * Ai * x]]
        push!(VtAVis, VtAVi)
    end
    VtAV = zeros(1,1)
    Vtbis = Vector{Float64}[]
    for bi in bis
        push!(Vtbis, Float64[x' * bi])
    end
    Vtb = zeros(1)
    approx_sol(p,VtAVis,Vtbis, VtAV, Vtb) = begin
        VtAV .= 0.0
        for i in eachindex(VtAVis)
            θAi = makeθAi(p,i)
            for j in eachindex(VtAVis[1])
                VtAV[:,j] .+= θAi .* VtAVis[i][j]
            end
        end
        Vtb .= 0.0
        for i in eachindex(Vtbis)
            Vtb .+= makeθbi(p,i) .* Vtbis[i]
        end
        return VtAV \ Vtb
    end
    return (V, VtAVis, VtAV, Vtbis, Vtb, approx_sol)
end

"""
`append_affine_rbm!(x, res_init, V, Ais, VtAVis, bis, Vtbis)`

Given a new vector `x` to append as a new column to `V`, update
the matrices in `VtAVis` and the vectors in `Vtbis`. Also, return
`VtAV` and `Vtb` which are preallocated, properly sized, matrices.
"""
function append_affine_rbm!(x::AbstractVector, V::AbstractVector, Ais::AbstractVector, 
                            VtAVis::AbstractVector, bis::AbstractVector, Vtbis::AbstractVector)
    for i in eachindex(Ais)
        # Add to each row of VtAVis[i]
        for k in eachindex(V)
            newrowval = x' * Ais[i] * V[k]
            push!(VtAVis[i][k], newrowval)
        end
        newcol = zeros(length(V)+1)
        # Add to new column
        for k in eachindex(V)
            newcol[k] = V[k]' * Ais[i] * x
        end
        newcol[end] = x' * Ais[i] * x
        push!(VtAVis[i], newcol)
    end
    VtAV = zeros(length(V)+1,length(V)+1)
    for i in eachindex(bis)
        Vtbis_end = x' * bis[i]
        push!(Vtbis[i], Vtbis_end)
    end
    Vtb = zeros(length(V)+1)
    push!(V, x)
    return (VtAV, Vtb)
end

"""
`GreedyRBAffineLinear(param_disc, Ais, makeθAi, bis, makeθbi, approx_stability_factor[, ϵ=1e-2, param_disc=nothing; max_snapshots=-1, noise=1])`

Constructs a `Greedy_RB_Affine_Linear` object for parametrized problem `A(p) x(p) = b(p)` with affine parameter
dependence:
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.

`param_disc` must either be a matrix with columns as parameters, or a vector of parameter vectors.
Uses the function `approx_stability_factor` to approximate the stability factor for each parameter in the discretization, 
chooses first parameter to be one with smallest stability factor. Afterwards, utilizes the affine decomposition of `A(p)` 
and `b(p)` to compute the norm of the residual, and compute upper-bounds for the error:
`||x - V x_r|| <= ||b(p) - A(p) V x_r|| / σ_min(A(p))`.

Greedily chooses parameters and then forms truth solution for parameter that results in largest upper-bound
for error. Then computes `||x - V x_r||`, and loops, adding to reduced basis, `V`, until the truth error
is less than `ϵ`.

If `max_snapshots` specified, will halt after that many if `ϵ` accuracy not yet reached. `noise` determines amount 
of printed output, `0` for no output, `1` for basic, and `2` for more.
"""
function GreedyRBAffineLinear(param_disc::Union{<:AbstractMatrix,<:AbstractVector},
                              Ais::AbstractVector,
                              makeθAi::Function,
                              bis::AbstractVector,
                              makeθbi::Function,
                              approx_stability_factor::Function,
                              ϵ=1e-2;
                              max_snapshots=-1,
                              noise=1)
    if param_disc isa AbstractMatrix
        params = eachcol(param_disc)
    else
        params = param_disc
    end
    # Choose a parameter vector p to begin with
    Atruth = zeros(size(Ais[1]))
    btruth = zeros(length(bis[1]))
    truth_sol(p) = begin
        Atruth .= 0.0
        for i in eachindex(Ais)
            Atruth .+= makeθAi(p,i) .* Ais[i]
        end
        btruth .= 0.0
        for i in eachindex(bis)
            btruth .+= makeθbi(p,i) .* bis[i]
        end
        return Atruth \ btruth
    end
    # Begin iteration by parameter with minimum stability factor
    if noise >= 1
        @printf("Beginning greedy selection, looping until truth error less than ϵ=%.2e",ϵ)
        if max_snapshots >= 1
            @printf("\nor until %d snapshots formed", max_snapshots)
        end
        print("\n----------\n")
    end
    maxerr = 0
    p1 = params[1]
    for p in params
        stability_factor = approx_stability_factor(p)
        err = 1.0 / stability_factor
        if err > maxerr
            p1 = p
            maxerr = err
        end
    end
    x = truth_sol(p1)
    x = x ./ norm(x)
    V, VtAVis, VtAV, Vtbis, Vtb, approx_sol = init_affine_rbm(x, Ais, bis, makeθAi, makeθbi)
    # Initialize res_init to compute residual norm
    res_init = residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, reshape(x, (length(x),1)))
    if noise >= 1
        print("k=1, first parameter value chosen, initialized for greedy search\n")
    end
    # Greedily loop over every parameter value
    ps = [p1]
    if max_snapshots == -1
        max_snapshots = length(params)
    end
    k = 1
    while k < max_snapshots
        k += 1
        maxerr = 0
        maxp = nothing
        for p in params
            if p in ps
                continue
            end
            stability_factor = approx_stability_factor(p)
            x_r = approx_sol(p,VtAVis,Vtbis,VtAV,Vtb)
            res_norm = residual_norm_affine_online(res_init, x_r, p)
            err = res_norm / stability_factor
            if err > maxerr
                maxp = p
                maxerr = err
            end
        end
        # Compute full solution to add to V
        x = truth_sol(maxp)
        x_r = approx_sol(maxp,VtAVis,Vtbis,VtAV,Vtb)
        x_approx = zeros(length(x))
        for i in eachindex(V)
            x_approx .+= x_r[i] .* V[i]
        end
        trueerr = norm(x .- x_approx)
        # Break condition
        if trueerr < ϵ
            if noise >= 1
                @printf("k=%d, truth error = %.4e, upperbound error = %.4e\n",k,trueerr,maxerr)
            end
            break
        end
        # Make orthogonal to v1,...,vn - Modified Gram-Schmidt
        for i in eachindex(V)
            x .= x .- dot(x, V[i]) .* V[i]
        end
        x .= x ./ norm(x)
        VtAV, Vtb = append_affine_rbm!(x, V, Ais, VtAVis, bis, Vtbis)
        add_col_to_V(res_init, x)
        push!(ps, maxp)
        if noise >= 1
            @printf("k=%d, truth error = %.4e, upperbound error = %.4e\n",k,trueerr,maxerr)
        end
    end
    if noise >= 1
        @printf("----------\nCompleted greedy selection after k=%d iterations\n",k)
    end
    return Greedy_RB_Affine_Linear(approx_stability_factor, res_init, Ais, 
                                   makeθAi, bis, makeθbi, params, ps, V, 
                                   VtAVis, Vtbis, ϵ, VtAV, Vtb)
end

"""
`GreedyRBAffineLinear(scm_init, Ais, makeθAi, bis, makeθbi[, ϵ=1e-2, param_disc=nothing; max_snapshots=-1, noise=1, sigma_eps=1.0, lb_stability=true])`

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

`sigma_eps` determines the relative error allowed between the upper and lower-bound approximations for the
stability factor. If the relative error is found to be greater than `sigma_eps`, then the stability factor is
explicitly computed and added to the `scm_init` object.

`lb_stability` defaults to true, and if true, then the lower bound for the stability factor is used
in the greedy method to ensure sharpness of the upper-bound on the error. If set to false, then will
use the upper-bound approximate of the stability factor instead.
"""
function GreedyRBAffineLinear(scm_init::SCM_Init,
                              Ais::AbstractVector,
                              makeθAi::Function,
                              bis::AbstractVector,
                              makeθbi::Function,
                              ϵ=1e-2,
                              param_disc::Union{Matrix,Vector,Nothing}=nothing;
                              max_snapshots=-1,
                              noise=1,
                              sigma_eps=1.0,
                              lb_stability=true)
    if isnothing(param_disc)
        params = scm_init.tree.data
    elseif param_disc isa Matrix
        params = eachcol(param_disc)
    else
        params = param_disc
    end
    approx_stability_factor(p) = find_sigma_bounds(scm_init, p, sigma_eps)[lb_stability ? 1 : 2]
    return GreedyRBAffineLinear(params, Ais, makeθAi, bis, makeθbi, approx_stability_factor, ϵ, max_snapshots=max_snapshots, noise=noise)
end

"""
`greedy_rb_err_data(param_disc,Ais,makeθAi,bis,makeθbi,approx_stability_factor[,num_snapshots=10,ϵ=1e-2;noise=1)`

Generates error data for parametrized problem `A(p) x(p) = b(p)` with affine parameter
dependence:
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.

**Note**: This method calls the full order solver on all parameters in `param_disc`, may
take a long time to run.

See the docs for `GreedyRBAffineLinear` for more information.

Returns a dictionary, `ret_data`, with the following components:

`ret_data[:basis_dim]` - A vector of reduced basis dimensions from 2 to `num_snapshots`

`ret_data[:weak_greedy_ub]` - A vector of the maximum upperbound l2 error found by the 
(weak) greedy reduced basis method (see `GreedyRBAffineLinear`)

`ret_data[:weak_greedy_true]` - A vector of the maximum true l2 error found by the 
(weak) greedy reduced basis method (see `GreedyRBAffineLinear`)

`ret_data[:strong_greedy_err]` - A vector of the maximum l2 error found by a  
strong greedy reduced basis method, uses knowledge of all solutions

`ret_data[:strong_greedy_proj]` - A vector of the maximum l2 error found by projecting  
truth solutions onto the strong greedy reduced basis, uses knowledge of all solutions

`ret_data[:pca_err]` - A vector of the maximum l2 error found by a PCA/POD
reduced basis method, uses knowledge of all solutions

`ret_data[:strong_greedy_proj]` - A vector of the maximum l2 error found by projecting  
truth solutions onto the PCA/POD reduced basis, uses knowledge of all solutions
"""
function greedy_rb_err_data(param_disc::Union{<:AbstractMatrix,<:AbstractVector},
                            Ais::AbstractVector,
                            makeθAi::Function,
                            bis::AbstractVector,
                            makeθbi::Function,
                            approx_stability_factor::Function,
                            num_snapshots=10;
                            noise=1)
    if param_disc isa AbstractMatrix
        params = eachcol(param_disc)
    else
        params = param_disc
    end

    # Return data 
    ret_data = Dict(
        :basis_dim => 2:num_snapshots,
        :weak_greedy_ub => Float64[],
        :weak_greedy_true => Float64[],
        :strong_greedy_err => Float64[],
        :strong_greedy_proj => Float64[],
        :pca_err => Float64[],
        :pca_proj => Float64[]
    )

    # Generate truth solutions
    Atruth = zeros(size(Ais[1]))
    btruth = zeros(length(bis[1]))
    truth_sol(p) = begin
        Atruth .= 0.0
        for i in eachindex(Ais)
            Atruth .+= makeθAi(p,i) .* Ais[i]
        end
        btruth .= 0.0
        for i in eachindex(bis)
            btruth .+= makeθbi(p,i) .* bis[i]
        end
        return Atruth \ btruth
    end
    if noise >= 1
        println("Beginning computation of all truth solutions")
    end
    truth_sols = Vector{Float64}[]
    for p in params
        push!(truth_sols, truth_sol(p))
    end
    truth_sols_mat = reduce(hcat, truth_sols)
    if noise >= 1
        println("Completed computation of all truth solutions")
    end
    
    # Weak greedy algorithm
    maxerr_ub = 0
    maxpi = 0
    for (i,p) in enumerate(params)
        stability_factor = approx_stability_factor(p)
        err = 1.0 / stability_factor
        if err > maxerr_ub
            maxpi = i
            maxerr_ub = err
        end
    end
    x = truth_sols[maxpi]
    x = x ./ norm(x)
    V1, VtAVis1, VtAV1, Vtbis1, Vtb1, approx_sol1 = ModelOrderReductionToolkit.init_affine_rbm(x, Ais, bis, makeθAi, makeθbi)
    res_init1 = residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, reshape(x, (length(x),1)))
    # Strong greedy algorithm
    qr_proj = qr_projector(truth_sols_mat, num_snapshots)
    x = qr_proj.M[:,1]
    V2, VtAVis2, VtAV2, Vtbis2, Vtb2, approx_sol2 = ModelOrderReductionToolkit.init_affine_rbm(x, Ais, bis, makeθAi, makeθbi)
    # POD/PCA
    pca_proj = pca_projector(truth_sols_mat, num_snapshots)
    x = pca_proj.M[:,1]
    V3, VtAVis3, VtAV3, Vtbis3, Vtb3, approx_sol3 = ModelOrderReductionToolkit.init_affine_rbm(x, Ais, bis, makeθAi, makeθbi)
    
    if noise >= 1
        print("k=1, first parameters chosen, beginning looping\n")
    end
    # Greedily loop over every parameter value
    ps = [params[maxpi]]
    k = 1
    while k < num_snapshots
        k += 1
        # Weak greedy algorithm
        maxerr_ub = 0
        maxp = nothing
        maxerr_truth = 0
        for (i,p) in enumerate(params)
            if p in ps
                continue
            end
            stability_factor = approx_stability_factor(p)
            x_r = approx_sol1(p,VtAVis1,Vtbis1,VtAV1,Vtb1)
            res_norm = residual_norm_affine_online(res_init1, x_r, p)
            err = res_norm / stability_factor
            if err > maxerr_ub
                maxp = p
                maxerr_ub = err
            end
            x_approx = zeros(length(x))
            for i in eachindex(V1)
                x_approx .+= x_r[i] .* V1[i]
            end
            trutherr = norm(truth_sols[i] .- x_approx)
            if trutherr > maxerr_truth
                maxerr_truth = trutherr
            end
        end
        # Add to ret_data
        push!(ret_data[:weak_greedy_ub], maxerr_ub)
        push!(ret_data[:weak_greedy_true], maxerr_truth)
        # Compute full solution to add to V
        x = truth_sol(maxp)
        x_r = approx_sol1(maxp,VtAVis1,Vtbis1,VtAV1,Vtb1)
        # Make orthogonal to v1,...,vn - Modified Gram-Schmidt
        for i in eachindex(V1)
            x .= x .- dot(x, V1[i]) .* V1[i]
        end
        x .= x ./ norm(x)
        VtAV1, Vtb1 = ModelOrderReductionToolkit.append_affine_rbm!(x, V1, Ais, VtAVis1, bis, Vtbis1)
        add_col_to_V(res_init1, x)
        push!(ps, maxp)
        
        # Strong greedy algorithm
        maxerr_truth = 0
        maxerr_proj = 0
        for (i,p) in enumerate(params)
            x_r = approx_sol2(p,VtAVis2,Vtbis2,VtAV2,Vtb2)
            x_approx = zeros(length(x))
            for i in eachindex(V2)
                x_approx .+= x_r[i] .* V2[i]
            end
            trutherr = norm(truth_sols[i] .- x_approx)
            if trutherr > maxerr_truth
                maxerr_truth = trutherr
            end
            M = qr_proj.M[:,1:(k-1)]
            x_proj = M * (M' * truth_sols[i])
            projerr = norm(truth_sols[i] .- x_proj)
            if projerr > maxerr_proj
                maxerr_proj = projerr
            end
        end
        # Add to ret_data
        push!(ret_data[:strong_greedy_err], maxerr_truth)
        push!(ret_data[:strong_greedy_proj], maxerr_proj)
        # Append to matrices and vectors
        VtAV2, Vtb2 = ModelOrderReductionToolkit.append_affine_rbm!(qr_proj.M[:,k], V2, Ais, VtAVis2, bis, Vtbis2)
        
        # PCA/POD Algorithm
        maxerr_truth = 0
        maxerr_proj = 0
        for (i,p) in enumerate(params)
            x_r = approx_sol3(p,VtAVis3,Vtbis3,VtAV3,Vtb3)
            x_approx = zeros(length(x))
            for i in eachindex(V3)
                x_approx .+= x_r[i] .* V3[i]
            end
            trutherr = norm(truth_sols[i] .- x_approx)
            if trutherr > maxerr_truth
                maxerr_truth = trutherr
            end
            M = pca_proj.M[:,1:(k-1)]
            x_proj = M * (M' * truth_sols[i])
            projerr = norm(truth_sols[i] .- x_proj)
            if projerr > maxerr_proj
                maxerr_proj = projerr
            end
        end
        # Add to ret_data
        push!(ret_data[:pca_err], maxerr_truth)
        push!(ret_data[:pca_proj], maxerr_proj)
        # Append to matrices and vectors
        VtAV3, Vtb3 = ModelOrderReductionToolkit.append_affine_rbm!(pca_proj.M[:,k], V3, Ais, VtAVis3, bis, Vtbis3)

        if noise >= 1
            @printf("Completed dimension k=%d\n",k)
        end

    end
    if noise >= 1
        @printf("----------\nCompleted after k=%d iterations, returning\n",k)
    end
    return ret_data
end