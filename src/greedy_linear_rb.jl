"""
`Greedy_RB_Affine_Linear`

Struct for containing initialized greedy reduced-basis method
for parametrized problem A(p) u = b(p) with affine parameter
dependence
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.

Uses Galerkin projection onto span of columns of `V`, 
`V' A(p) V u_r = V' b(p)`, with `V u_r ≈ u = A(p)^(-1) b(p)`.

Given a new parameter vector, `p`, and an object
`greedy_sol::Greedy_RB_Affine_Linear`, form the reduced
basis solution with `greedy_sol(p[, full=true])`. 
"""
struct Greedy_RB_Affine_Linear
    approx_stability_factor::Function
    compute_res_norm::Function
    update_res_norm!::Function
    Ais::AbstractVector
    makeθAi::Function
    bis::AbstractVector
    makeθbi::Function
    params_greedy::AbstractVector
    V::Vector{Vector}
    VtAVis::Vector{Vector{Vector}}
    Vtbis::Vector{Vector}
    A_alloc::Matrix # Preallocated
    b_alloc::Vector
    ualloc1::Vector
    ualloc2::Vector
    maxerrs::Vector
    trutherrs::Vector
    T::Type
end

function Base.show(io::Core.IO, greedy_sol::Greedy_RB_Affine_Linear)
    res = "Initialized reduced basis method for parametrized problem A(p) x = b(p) with affine parameter dependence:\n"
    res *= "    A(p) = ∑ makeθAi(p,i) Ais[i],\n"
    res *= "    b(p) = ∑ makeθbi(p,i) bis[i].\n"
    res *= "Galerkin projection is used onto an $(length(greedy_sol.V)) dimensional space:\n"
    res *= "    V' A(p) V u_r = V' b(p),\n"
    res *= "    V u_r ≈ u = A(p)^(-1) b(p)."
    print(io, res)
end

"""
`(greedy_sol::Greedy_RB_Affine_Linear)(p[; full=true, truth=false])`

It `truth=false`, forms the reduced basis solution for the parameter vector `p`,
`(V' A(p) V)^(-1) V' b(p)`. If `full` is set to false, then returns
the shortened vector `(V' A(p))^(-1) V' b(p)`.

If `truth=true`, then forms the truth solution `A(p)^(-1) b(p)`.
"""
function (greedy_sol::Greedy_RB_Affine_Linear)(p; full=true, truth=false)
    if truth
        A = greedy_sol.A_alloc
        A .= 0
        for i in eachindex(greedy_sol.Ais)
            A .+= greedy_sol.makeθAi(p,i) .* greedy_sol.Ais[i]
        end
        b = greedy_sol.b_alloc
        b .= 0
        for i in eachindex(greedy_sol.bis)
            b .+= greedy_sol.makeθbi(p,i) .* greedy_sol.bis[i]
        end
        return A \ b
    end
    r = length(greedy_sol.V)
    if r == 0
        # Basis is empty, solution is zero vector
        if full
            return zeros(greedy_sol.T, size(greedy_sol.A_alloc,1))
        else
            return zeros(greedy_sol.T, 0)
        end
    end
    VtAV_r = view(greedy_sol.A_alloc, 1:r, 1:r)
    VtAV_r .= 0.0
    for i in eachindex(greedy_sol.VtAVis)
        θAi = greedy_sol.makeθAi(p,i)
        for j in eachindex(greedy_sol.VtAVis[1])
            VtAV_r[:,j] .+= θAi .* greedy_sol.VtAVis[i][j]
        end
    end
    Vtb_r = view(greedy_sol.b_alloc, 1:r)
    Vtb_r .= 0.0
    for i in eachindex(greedy_sol.Vtbis)
        Vtb_r .+= greedy_sol.makeθbi(p,i) .* greedy_sol.Vtbis[i]
    end
    u_r = VtAV_r \ Vtb_r
    if full
        u_approx = zeros(eltype(u_r),length(greedy_sol.V[1]))
        for i in eachindex(u_r)
            u_approx .+= u_r[i] * greedy_sol.V[i]
        end
        return u_approx
    end
    return u_r
end

""" TODO Update header
`init_affine_rbm(x, Ais, bis, makeθAi, makeθbi, T)`

Given a vector `x`, generates the matrices `V` and `VtAVis`,
and the vectors `Vtbis` for preallocation and quick computation
of reduced basis solutions for the problem `A(p)x(p)=b(p)` with affinely
dependent matrix `A(p) = ∑ makeθAi(p,i) Ais[i]` and vector
`b(p) = ∑ makeθbi(p,i) bis[i]`. Must pass in type `T`.
"""
function init_affine_rbm(Ais::AbstractVector, bis::AbstractVector, T::Type)
    # Preallocate vectors and matrices
    A_alloc = all(issparse.(Ais)) ? spzeros(T,size(Ais[1])) : zeros(T,size(Ais[1]))
    b_alloc = zeros(T,length(bis[1]))
    ualloc1 = zeros(T,length(bis[1]))
    ualloc2 = zeros(T,length(bis[1]))
    # Form Galerkin projection matrices
    V = Vector{T}[]
    VtAVis = Vector{Vector{T}}[]
    for _ in Ais
        VtAVi = Vector{T}[]#T[]]
        push!(VtAVis, VtAVi)
    end
    Vtbis = Vector{T}[]
    for _ in bis
        push!(Vtbis, T[])
    end
    ps_greedy = []
    return (A_alloc, b_alloc, ualloc1, ualloc2, V, VtAVis, Vtbis, ps_greedy)
end

"""
`append_affine_rbm!(greedy_sol, x)`

Given a `Greedy_RB_Affine_Linear` object, `greedy_sol`, and a new 
vector `x`, update the vectors and matrices in `greedy_sol`.
"""
function append_affine_rbm!(greedy_sol::Greedy_RB_Affine_Linear, x::AbstractVector; noise=1)
    for i in eachindex(greedy_sol.Ais)
        # Add to each row of VtAVis[i]
        for k in eachindex(greedy_sol.V)
            newrowval = x' * greedy_sol.Ais[i] * greedy_sol.V[k]
            push!(greedy_sol.VtAVis[i][k], newrowval)
        end
        newcol = zeros(greedy_sol.T,length(greedy_sol.V)+1)
        # Add to new column
        for k in eachindex(greedy_sol.V)
            newcol[k] = greedy_sol.V[k]' * greedy_sol.Ais[i] * x
        end
        newcol[end] = x' * greedy_sol.Ais[i] * x
        push!(greedy_sol.VtAVis[i], newcol)
    end
    for i in eachindex(greedy_sol.bis)
        Vtbis_end = x' * greedy_sol.bis[i]
        push!(greedy_sol.Vtbis[i], Vtbis_end)
    end
    push!(greedy_sol.V, x)
end

# TODO add docstring
function add_param_greedily!(greedy_method::Greedy_RB_Affine_Linear, params::Vector, ϵ=0; noise=1)
    k = length(greedy_method.params_greedy) + 1
    if k > size(greedy_method.A_alloc, 1)
        println("Cannot add more than $(k-1) snapshots, not appending to greedy_method object\n")
        return true
    end
    maxerr = -1.0
    maxp = nothing
    for p in params
        if p in greedy_method.params_greedy
            continue
        end
        stability_factor = greedy_method.approx_stability_factor(p)
        # Form reduced basis solution
        u_r = greedy_method(p, full=false, truth=false)
        res_norm = greedy_method.compute_res_norm(u_r, p)
        err = res_norm / stability_factor
        if err > maxerr
            maxp = p
            maxerr = err
        end
    end
    greedy_method.maxerrs[k] = maxerr
    # Break condition
    if maxerr < ϵ
        if noise >= 1
            @printf("k=%d, upperbound error = %.4e < ϵ, not appending to greedy_method object\n",k,maxerr)
        end
        return true
    end
    # Compute full solution to add to V
    u = greedy_method(maxp, truth=true)
    u_r = greedy_method(maxp, full=false)
    # Form approximate solution without allocating
    greedy_method.ualloc1 .= 0
    for (i,v) in enumerate(greedy_method.V)
        greedy_method.ualloc1 .+= u_r[i] .* v
    end
    trueerr = norm(u .- greedy_method.ualloc1)
    greedy_method.trutherrs[k] = trueerr
    # Orthonormalize u w.r.t. V
    nu = orthonormalize_mgs2!(u, greedy_method.V)
    if nu != 0
        append_affine_rbm!(greedy_method, u)
        greedy_method.update_res_norm!(u)
    elseif noise >= 1
        println("After orthonormalization, truth vector had norm 0, not appending to greedy_method object")
    end
    push!(greedy_method.params_greedy, maxp)
    if noise >= 1
        @printf("k=%d, truth error = %.4e, upperbound error = %.4e\n",k,trueerr,maxerr)
    end
    return false
end

# TODO update docstring
"""
`GreedyRBAffineLinear(param_disc, Ais, makeθAi, bis, makeθbi, approx_stability_factor[, ϵ=1e-2, param_disc=nothing; max_snapshots=-1, noise=1])`

Constructs a `Greedy_RB_Affine_Linear` object for parametrized problem `A(p) y(p) = b(p)` with affine parameter
dependence:
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.

`param_disc` must either be a matrix with columns as parameters, or a vector of parameter vectors.
Uses the function `approx_stability_factor` to approximate the stability factor for each parameter in the discretization, 
chooses first parameter to be one with smallest stability factor. Afterwards, utilizes the affine decomposition of `A(p)` 
and `b(p)` to compute the norm of the residual, and compute upper-bounds for the error:
`||u - V u_r|| <= ||b(p) - A(p) V u_r|| / σ_min(A(p))`.

Greedily chooses parameters and then forms truth solution for parameter that results in largest upper-bound
for error. Then computes `||u - V u_r||`, and loops, adding to reduced basis, `V`, until the truth error
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
                              compute_res_norm::Function,
                              update_res_norm!::Function,
                              ϵ=1e-2;
                              T::Type=Float64,
                              max_snapshots=-1,
                              noise=1)
    if param_disc isa AbstractMatrix
        params = eachcol(param_disc)
    else
        params = param_disc
    end
    if noise >= 1
        @printf("Beginning greedy selection, looping until upperbound error less than ϵ=%.2e",ϵ)
        if max_snapshots >= 1
            @printf("\nor until %d snapshots formed", max_snapshots)
        end
        print("\n----------\n")
    end
    A_alloc, b_alloc, ualloc1, ualloc2, V, VtAVis, Vtbis, ps_greedy = init_affine_rbm(Ais, bis, T)
    if max_snapshots == -1
        max_snapshots = size(A,1)
    end
    # Form error vectors 
    maxerrs = zeros(max_snapshots)
    trutherrs = zeros(max_snapshots)
    if noise >= 1
        print("Initialized for greedy search\n")
    end
    # Form greedy method object
    greedy_method = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ais, makeθAi, 
                                            bis, makeθbi, ps_greedy, V, VtAVis, 
                                            Vtbis, A_alloc, b_alloc, ualloc1,
                                            ualloc2, maxerrs, trutherrs, T)
    # Greedily loop over every parameter value
    for k in 1:max_snapshots
        hit_eps = add_param_greedily!(greedy_method, params, ϵ, noise=noise)
        if hit_eps
            break
        end
    end
    k = length(greedy_method.params_greedy)
    if noise >= 1
        @printf("----------\nCompleted greedy selection after k=%d iterations\n",k)
    end
    return greedy_method
end

#TODO Add docstring
function GreedyRBAffineLinear(param_disc::Union{<:AbstractMatrix,<:AbstractVector},
                              Ais::AbstractVector,
                              makeθAi::Function,
                              bis::AbstractVector,
                              makeθbi::Function,
                              approx_stability_factor::Function,
                              ϵ=1e-2;
                              res_calc::Int=2,
                              T::Type=Float64,
                              max_snapshots=-1,
                              noise=1)
    # Create compute_res_norm(u_r,p) and update_res_norm!(v) methods
    if res_calc==0
        ualloc1 = zeros(T, size(Ais[1],1))
        ualloc2 = zeros(T, size(Ais[1],1))
        V = Vector{T}[]
        AiVjs = Vector{Vector{T}}[]
        for i in eachindex(Ais)
            AiVj = Vector{T}[]
            for j in eachindex(V)
                push!(AiVj, Ais[i] * V[j])
            end
            push!(AiVjs, AiVj)
        end
        compute_res_norm = (u_r,p) -> begin 
            # Form A V u_r
            ualloc1 .= 0
            for i in eachindex(Ais)
                for j in eachindex(V)
                    ualloc1 .+= makeθAi(p,i) .* u_r[j] .* AiVjs[i][j]
                end
            end
            # Form b
            ualloc2 .= 0
            for i in eachindex(bis)
                ualloc2 .+= makeθbi(p,i) .* bis[i]
            end
            ualloc2 .-= ualloc1 # b - A V u_r
            return sqrt(real(dot(ualloc2,ualloc2)))
        end
        update_res_norm! = (v) -> begin 
            push!(V, v)
            for i in eachindex(Ais)
                push!(AiVjs[i], Ais[i] * v)
            end
        end
    elseif res_calc == 1
        # Initialize res_init to compute residual norm
        res_init = residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, Vector{T}[], T=T)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v, T)
    else
        # Initialize res_init to compute residual norm
        res_init = residual_norm_affine_proj_init(Ais, makeθAi, bis, makeθbi, Vector{T}[], T=T)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v, T)
    end
    return GreedyRBAffineLinear(param_disc, Ais, makeθAi, bis, makeθbi, approx_stability_factor, compute_res_norm, update_res_norm!, ϵ, T=T, max_snapshots=max_snapshots, noise=noise)
end

"""
`greedy_rb_err_data(param_disc,Ais,makeθAi,bis,makeθbi,approx_stability_factor[,num_snapshots=10,ϵ=1e-2;noise=1)`

Generates error data for parametrized problem `A(p) u(p) = b(p)` with affine parameter
dependence:
`A(p) = ∑ makeθAi(p,i) Ais[i]`, and
`b(p) = ∑ makeθbi(p,i) bis[i]`.

**Note**: This method calls the full order solver on all parameters in `param_disc`, may
take a long time to run.

See the docs for `GreedyRBAffineLinear` for more information.

Returns a dictionary, `ret_data`, with the following components:

`ret_data[:truth_sols]` = Vector containing all truth solutions.

`ret_data[:basis_dim]` - A vector of reduced basis dimensions from 2 to `num_snapshots`

`ret_data[:weak_greedy_ub]` - A vector of the maximum upperbound l2 error found by the 
(weak) greedy reduced basis method (see `GreedyRBAffineLinear`)

`ret_data[:weak_greedy_true]` - A vector of the truth l2 error of the vector chosen by the 
(weak) greedy reduced basis method (see `GreedyRBAffineLinear`)

`ret_data[:weak_greedy_true_ub]` - A vector of the maximum true l2 error found by the 
(weak) greedy reduced basis method (see `GreedyRBAffineLinear`)

`ret_data[:strong_greedy_err]` - A vector of the maximum l2 error found by a  
strong greedy reduced basis method, uses knowledge of all solutions

`ret_data[:strong_greedy_proj]` - A vector of the maximum l2 error found by projecting  
truth solutions onto the strong greedy reduced basis, uses knowledge of all solutions

`ret_data[:qr_projector]` - A projection object which can be used to orthogonally project
vectors onto the strong greedy subspace. See ModelOrderReductionToolkit.qr_projector method.

`ret_data[:pca_err]` - A vector of the maximum l2 error found by a PCA/POD
reduced basis method, uses knowledge of all solutions

`ret_data[:strong_greedy_proj]` - A vector of the maximum l2 error found by projecting  
truth solutions onto the PCA/POD reduced basis, uses knowledge of all solutions

`ret_data[:pca_projector]` - A projection object which can be used to orthogonally project
vectors onto the SVD/PCA subspace. See ModelOrderReductionToolkit.pca_projector method.
"""
function greedy_rb_err_data(param_disc::Union{<:AbstractMatrix,<:AbstractVector},
                            Ais::AbstractVector,
                            makeθAi::Function,
                            bis::AbstractVector,
                            makeθbi::Function,
                            approx_stability_factor::Function,
                            num_snapshots=10;
                            res_calc::Int=2,
                            T::Type=Float64,
                            noise=1)
    if param_disc isa AbstractMatrix
        params = eachcol(param_disc)
    else
        params = param_disc
    end

    # Return data 
    ret_data = Dict{Symbol,Any}(
        :basis_dim => 1:num_snapshots,
        :weak_greedy_ub => Float64[],
        :weak_greedy_true => Float64[],
        :weak_greedy_true_ub => Float64[],
        :strong_greedy_err => Float64[],
        :strong_greedy_proj => Float64[],
        :pca_err => Float64[],
        :pca_proj => Float64[]
    )

    # Weak greedy algorithm
    A_alloc, b_alloc, ualloc1, ualloc2, V1, VtAVis1, Vtbis1, ps_greedy = init_affine_rbm(Ais, bis, T)
    # Create compute_res_norm(u_r,p) and update_res_norm!(v) methods
    if res_calc == 0
        AiVjs = Vector{Vector{T}}[]
        for i in eachindex(Ais)
            AiVj = Vector{T}[]
            for j in eachindex(V1)
                push!(AiVj, Ais[i] * V1[j])
            end
            push!(AiVjs, AiVj)
        end
        compute_res_norm = (u_r,p) -> begin 
            # Form A V u_r
            ualloc1 .= 0
            for i in eachindex(Ais)
                for j in eachindex(V1)
                    ualloc1 .+= makeθAi(p,i) .* u_r[j] .* AiVjs[i][j]
                end
            end
            # Form b
            ualloc2 .= 0
            for i in eachindex(bis)
                ualloc2 .+= makeθbi(p,i) .* bis[i]
            end
            ualloc2 .-= ualloc1 # b - A V u_r
            return sqrt(real(dot(ualloc2,ualloc2)))
        end
        update_res_norm! = (v) -> begin 
            push!(V1, v)
            for i in eachindex(Ais)
                push!(AiVjs[i], Ais[i] * v)
            end
        end
    elseif res_calc == 1
        # Initialize res_init to compute residual norm
        res_init = residual_norm_affine_init(Ais, makeθAi, bis, makeθbi, V1, T=T)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v, T)
    else
        # Initialize res_init to compute residual norm
        res_init = residual_norm_affine_proj_init(Ais, makeθAi, bis, makeθbi, Vector{T}[], T=T)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v, T)
    end
    # Form greedy method object
    greedy_method = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ais, makeθAi, 
                                            bis, makeθbi, ps_greedy, V1, VtAVis1, 
                                            Vtbis1, A_alloc, b_alloc, ualloc1,
                                            ualloc2, zeros(num_snapshots), zeros(num_snapshots), T)

    if noise >= 1
        println("Beginning computation of all truth solutions")
    end
    truth_sols = Vector{T}[]
    for p in params
        push!(truth_sols, greedy_method(p, truth=true))
    end
    truth_sols_mat = reduce(hcat, truth_sols)
    if noise >= 1
        println("Completed computation of all truth solutions")
    end
    ret_data[:truth_sols] = truth_sols

    # Strong greedy algorithm
    qr_proj = qr_projector(truth_sols_mat, num_snapshots)
    ret_data[:qr_projector] = qr_proj
    _, _, _, _, V2, VtAVis2, Vtbis2, ps_greedy2 = init_affine_rbm(Ais, bis, T)
    strong_greedy = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ais, makeθAi, 
                                            bis, makeθbi, ps_greedy2, V2, VtAVis2, 
                                            Vtbis2, A_alloc, b_alloc, ualloc1,
                                            ualloc2, [], [], T)
    # POD/PCA
    pca_proj = pca_projector(truth_sols_mat, num_snapshots)
    ret_data[:pca_projector] = pca_proj
    _, _, _, _, V3, VtAVis3, Vtbis3, ps_greedy3 = init_affine_rbm(Ais, bis, T)
    pca_method = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ais, makeθAi, 
                                            bis, makeθbi, ps_greedy3, V3, VtAVis3, 
                                            Vtbis3, A_alloc, b_alloc, ualloc1,
                                            ualloc2, [], [], T)
    
    # Greedily loop over every parameter value
    for k in 1:num_snapshots
        # Weak greedy algorithm
        maxerr_truth_ub = 0
        for (i,p) in enumerate(params)
            err_truth_ub = norm(truth_sols[i] .- greedy_method(p, full=true))
            if err_truth_ub > maxerr_truth_ub
                maxerr_truth_ub = err_truth_ub
            end
        end
        add_param_greedily!(greedy_method, params, 0, noise=noise-1)
        # Add to ret_data
        push!(ret_data[:weak_greedy_ub], greedy_method.maxerrs[k])
        push!(ret_data[:weak_greedy_true], greedy_method.trutherrs[k])
        push!(ret_data[:weak_greedy_true_ub], maxerr_truth_ub)
        
        # Strong greedy algorithm
        maxerr_truth_ub = 0
        maxerr_proj = 0
        for (i,p) in enumerate(params)
            u_approx = strong_greedy(p, full=true)
            trutherr = norm(truth_sols[i] .- u_approx)
            if trutherr > maxerr_truth_ub
                maxerr_truth_ub = trutherr
            end
            M = view(qr_proj.M,:,1:(k-1))
            u_proj = M * (M' * truth_sols[i])
            projerr = norm(truth_sols[i] .- u_proj)
            if projerr > maxerr_proj
                maxerr_proj = projerr
            end
        end
        # Add to ret_data
        push!(ret_data[:strong_greedy_err], maxerr_truth_ub)
        push!(ret_data[:strong_greedy_proj], maxerr_proj)
        # Append to matrices and vectors
        append_affine_rbm!(strong_greedy, qr_proj.M[:,k])
        
        # PCA/POD Algorithm
        maxerr_truth_ub = 0
        maxerr_proj = 0
        for (i,p) in enumerate(params)
            u_approx = pca_method(p, full=true)
            trutherr = norm(truth_sols[i] .- u_approx)
            if trutherr > maxerr_truth_ub
                maxerr_truth_ub = trutherr
            end
            M = view(pca_proj.M,:,1:(k-1))
            u_proj = M * (M' * truth_sols[i])
            projerr = norm(truth_sols[i] .- u_proj)
            if projerr > maxerr_proj
                maxerr_proj = projerr
            end
        end
        # Add to ret_data
        push!(ret_data[:pca_err], maxerr_truth_ub)
        push!(ret_data[:pca_proj], maxerr_proj)
        # Append to matrices and vectors
        append_affine_rbm!(pca_method, pca_proj.M[:,k])
        if noise >= 1
            println("Completed dimension k=$(k)")
        end
    end
    return ret_data
end