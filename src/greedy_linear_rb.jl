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
struct Greedy_RB_Affine_Linear{T}
    approx_stability_factor::Function
    compute_res_norm::Function
    update_res_norm!::Function
    Ap::APArray
    bp::APArray
    params_greedy::AbstractVector
    V::VectorOfVectors{T}
    VtAVp::APArray
    Vtbp::APArray
    A_alloc::AbstractMatrix{T} # Preallocated
    b_alloc::Vector{T}
    ualloc::Vector{T}
    maxerrs::Vector{Float64}
    trutherrs::Vector{Float64}
end

function Base.show(io::Core.IO, greedy_sol::Greedy_RB_Affine_Linear)
    res = "Initialized reduced basis method for parametrized problem A(p) x = b(p) with affine parameter dependence:\n"
    res *= "    A(p) = ∑ makeθAi(p,i) Ais[i],\n"
    res *= "    b(p) = ∑ makeθbi(p,i) bis[i].\n"
    res *= "Galerkin projection is used onto an $(size(greedy_sol.V,2)) dimensional space:\n"
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
function (greedy_sol::Greedy_RB_Affine_Linear{T})(p; full=true, truth=false) where T
    if truth
        A = greedy_sol.A_alloc
        formArray!(greedy_sol.Ap, A, p)
        b = greedy_sol.b_alloc
        formArray!(greedy_sol.bp, b, p)
        return A \ b
    end
    r = size(greedy_sol.V,2)
    if r == 0
        # Basis is empty, solution is zero vector
        if full
            return zeros(T, size(greedy_sol.A_alloc,1))
        else
            return zeros(T, 0)
        end
    end
    VtAV_r = view(greedy_sol.A_alloc, 1:r, 1:r)
    formArray!(greedy_sol.VtAVp, VtAV_r, p)
    Vtb_r = view(greedy_sol.b_alloc, 1:r)
    formArray!(greedy_sol.Vtbp, Vtb_r, p)
    u_r = VtAV_r \ Vtb_r
    if full
        u_approx = greedy_sol.V * u_r
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
function init_affine_rbm(Ap::APArray, 
                         bp::APArray,
                         T::Type)
    Ais = Ap.arrays
    bis = bp.arrays
    # Preallocate vectors and matrices
    A_alloc = all(issparse.(Ais)) ? spzeros(T,size(Ais[1])) : zeros(T,size(Ais[1]))
    b_alloc = zeros(T,length(bis[1]))
    ualloc = zeros(T,length(bis[1]))
    # Form Galerkin projection matrices
    V = VectorOfVectors(size(Ais[1],1),0,T)
    VtAVis = VectorOfVectors{T}[]
    for _ in Ais
        VtAVi = VectorOfVectors(0,0,T)
        push!(VtAVis, VtAVi)
    end
    VtAVp = APArray(VtAVis, Ap.makeθi)
    Vtbis = Vector{T}[]
    for _ in bis
        push!(Vtbis, T[])
    end
    Vtbp = APArray(Vtbis, bp.makeθi)
    ps_greedy = []
    return (A_alloc, b_alloc, ualloc, V, VtAVp, Vtbp, ps_greedy)
end

"""
`append_affine_rbm!(greedy_sol, x)`

Given a `Greedy_RB_Affine_Linear` object, `greedy_sol`, and a new 
vector `x`, update the vectors and matrices in `greedy_sol`.
"""
function append_affine_rbm!(greedy_sol::Greedy_RB_Affine_Linear, x::AbstractVector; noise=1)
    Ais = greedy_sol.Ap.arrays
    bis = greedy_sol.bp.arrays
    for i in eachindex(Ais)
        VtAVi = greedy_sol.VtAVp.arrays[i]
        nrows,ncols = size(VtAVi)
        # Add row to VtAVis[i]
        addRow!(VtAVi)
        if ncols > 0
            VtAVi[end:end,:] .= (x' * Ais[i]) * greedy_sol.V 
        end
        addCol!(VtAVi)        # Add to new column
        if nrows > 0
            VtAVi[1:end-1,end:end] .= greedy_sol.V' * (Ais[i] * x)
        end
        VtAVi[end,end] = x' * Ais[i] * x
    end
    for i in eachindex(bis)
        Vtbis_end = x' * bis[i]
        push!(greedy_sol.Vtbp.arrays[i], Vtbis_end)
    end
    addCol!(greedy_sol.V, x)
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
    push!(greedy_method.maxerrs, maxerr)
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
    # Form approximate solution
    if k == 1
        greedy_method.ualloc .= 0
    else
        greedy_method.ualloc .= greedy_method.V * u_r
    end
    trueerr = norm(u .- greedy_method.ualloc)
    push!(greedy_method.trutherrs, trueerr)
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

"""
`GreedyRBAffineLinear(param_disc, Ap, bp, approx_stability_factor, compute_res_norm, update_res_norm![, ϵ=1e-2; T=Float64, max_snapshots=-1, noise=1])`

Constructs a `Greedy_RB_Affine_Linear` object for parametrized problem `Ap(p) u(p) = bp(p)` with affine parameter
dependence given by APArrays `Ap` (matrix) and `bp` (vector)

`param_disc` must either be a matrix with columns as parameters, or a vector of parameters.
Uses the function `approx_stability_factor(p)` to approximate the stability factor for each parameter in the discretization,
and `compute_res_norm(u_r,p)` to approximate the norm of the residual. Utilizes the affine decomposition of `A(p)` 
and `b(p)` to compute upper-bounds for the error:
`||u - V u_r|| <= ||b(p) - A(p) V u_r|| / σ_min(A(p)) = compute_res_norm(u_r,p) / approx_stability_factor(p)`.
After the parameter value with the highest upperbound error is selected, its truth solution is computed, and it
is added to the reduced basis. The method `update_res_norm!(u)` is used to update the residual norm computation.

This procedure loops until the upper-bound error is less than `ϵ`, or until the number of snapshots exceeds `max_snapshots`,
or the minimum of number of parameters and number of columns of `A` by default.

Specify `T` if wish to use number type other than `Float64`, works for `ComplexF64`.

`noise` determines amount of printed output, `0` for no output, `1` for basic, and `2` for more.
"""
function GreedyRBAffineLinear(param_disc::Union{<:AbstractMatrix,<:AbstractVector},
                              Ap::APArray,
                              bp::APArray,
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
    A_alloc, b_alloc, ualloc, V, VtAVp, Vtbp, ps_greedy = init_affine_rbm(Ap, bp, T)
    if max_snapshots == -1
        max_snapshots = size(Ap.arrays[1],1)
    end
    # Form error vectors 
    maxerrs = Float64[]
    trutherrs = Float64[]
    if noise >= 1
        print("Initialized for greedy search\n")
    end
    # Form greedy method object
    greedy_method = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ap, bp, ps_greedy, V, 
                                            VtAVp, Vtbp, A_alloc, b_alloc, ualloc,
                                            maxerrs, trutherrs)
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

# Deprecated for call with APArray objects
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
    Ap = APArray(Ais, makeθAi)
    bp = APArray(bis, makeθbi)
    return GreedyRBAffineLinear(param_disc, Ap, bp, approx_stability_factor, 
                                compute_res_norm, update_res_norm!, ϵ, T=T, 
                                max_snapshots=max_snapshots, noise=noise)
end

"""
`GreedyRBAffineLinear(param_disc, Ap, bp, approx_stability_factor[, ϵ=1e-2; res_calc=2, T=Float64, max_snapshots=-1, noise=1])`

Constructs a `Greedy_RB_Affine_Linear` object for parametrized problem `Ap(p) u(p) = bp(p)` with affine parameter
dependence given by APArrays `Ap` (matrix) and `bp` (vector).

`param_disc` must either be a matrix with columns as parameters, or a vector of parameter vectors.

Uses `res_calc` to determine how the norm of the residual `||b(p) - A(p) V u_r||` is computed. By default set to `2` which
uses a projection method and has online runtime independent of the dimension of `A`. If set to `1`, uses a method that also 
has online runtime independent of the dimension of `A`, but may become numerically unstable for large reduced bases. If `0`,
computes the residual norm explicitly which scales like `O(N)` where `N` is the dimension of `A`.

Uses the function `approx_stability_factor(p)` to approximate the stability factor for each parameter in the discretization,
and the above selection to compute the norm of the residual. Utilizes the affine decomposition of `A(p)` 
and `b(p)` to compute upper-bounds for the error:
`||u - V u_r|| <= ||b(p) - A(p) V u_r|| / σ_min(A(p)) = compute_res_norm(u_r,p) / approx_stability_factor(p)`.
After the parameter value with the highest upperbound error is selected, its truth solution is computed, and it
is added to the reduced basis. The method `update_res_norm!(u)` is used to update the residual norm computation.

This procedure loops until the upper-bound error is less than `ϵ`, or until the number of snapshots exceeds `max_snapshots`,
or the minimum of number of parameters and number of columns of `A` by default.

Specify `T` if wish to use number type other than `Float64`, works for `ComplexF64`.

`noise` determines amount of printed output, `0` for no output, `1` for basic, and `2` for more.
"""
function GreedyRBAffineLinear(param_disc::Union{<:AbstractMatrix,<:AbstractVector},
                              Ap::APArray,
                              bp::APArray,
                              approx_stability_factor::Function,
                              ϵ=1e-2;
                              res_calc::Int=2,
                              T::Type=Float64,
                              max_snapshots=-1,
                              noise=1)
    Ais = Ap.arrays
    makeθAi = Ap.makeθi
    bis = bp.arrays
    makeθbi = bp.makeθi
    # Create compute_res_norm(u_r,p) and update_res_norm!(v) methods
    if res_calc==0
        ualloc1 = zeros(T, size(Ais[1],1))
        ualloc2 = zeros(T, size(Ais[1],1))
        V = VectorOfVectors(size(Ais[1],1), 0, T)
        AiVjs = VectorOfVectors{T}[]
        for i in eachindex(Ais)
            AiVj = VectorOfVectors(Ais[i] * V)
            push!(AiVjs, AiVj)
        end
        compute_res_norm = (u_r,p) -> begin 
            # Form A V u_r
            ualloc1 .= 0
            for i in eachindex(Ais)
                ualloc1 .+= makeθAi(p,i) .* AiVjs[i] * u_r 
            end
            # Form b
            formArray!(bp, ualloc2, p)
            ualloc2 .-= ualloc1 # b - A V u_r
            return sqrt(real(dot(ualloc2,ualloc2)))
        end
        update_res_norm! = (v) -> begin 
            addCol!(V, v)
            for i in eachindex(Ais)
                addCol!(AiVjs[i], Ais[i] * v)
            end
        end
    elseif res_calc == 1
        # Initialize res_init to compute residual norm
        V = VectorOfVectors(size(Ais[1],1), 0, T)
        res_init = residual_norm_affine_init(Ap, bp, V)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v)
    else
        # Initialize res_init to compute residual norm
        V = VectorOfVectors(size(Ais[1],1), 0, T)
        res_init = residual_norm_affine_proj_init(Ap, bp, V)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v)
    end
    return GreedyRBAffineLinear(param_disc, Ais, makeθAi, bis, makeθbi, approx_stability_factor, compute_res_norm, update_res_norm!, ϵ, T=T, max_snapshots=max_snapshots, noise=noise)
end

# Deprecated for call with APArray objects
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
    Ap = APArray(Ais, makeθAi)
    bp = APArray(bis, makeθbi)
    return GreedyRBAffineLinear(param_disc, Ap, bp, approx_stability_factor, ϵ, res_calc=res_calc, 
                                T=T, max_snapshots=max_snapshots, noise=noise)
end

"""
`greedy_rb_err_data(param_disc,Ap,bp,approx_stability_factor[,num_snapshots=10,ϵ=1e-2;noise=1)`

Constructs a `Greedy_RB_Affine_Linear` object for parametrized problem `Ap(p) u(p) = bp(p)` with affine parameter
dependence given by APArrays `Ap` (matrix) and `bp` (vector).

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
                            Ap::APArray,
                            bp::APArray,
                            approx_stability_factor::Function,
                            num_snapshots=10;
                            res_calc::Int=2,
                            T::Type=Float64,
                            noise=1)
    Ais = Ap.arrays
    bis = bp.arrays
    
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
    A_alloc, b_alloc, ualloc1, V1, VtAVp1, Vtbp1, ps_greedy = init_affine_rbm(Ap, bp, T)
    ualloc2 = copy(ualloc1)
    # Create compute_res_norm(u_r,p) and update_res_norm!(v) methods
    if res_calc == 0
        N = size(Ais[1], 1)
        ualloc1 = zeros(T, N)
        ualloc2 = zeros(T, N)
        V = VectorOfVectors(N, 0, T)
        AiVjs = VectorOfVectors{T}[]
        for i in eachindex(Ais)
            AiVj = VectorOfVectors(Ais[i] * V)
            push!(AiVjs, AiVj)
        end
        compute_res_norm = (u_r,p) -> begin 
            # Form A V u_r
            ualloc1 .= 0
            for i in eachindex(Ais)
                ualloc1 .+= makeθAi(p,i) .* AiVjs[i] * u_r 
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
            addCol!(V, v)
            for i in eachindex(Ais)
                addCol!(AiVjs[i], Ais[i] * v)
            end
        end
    elseif res_calc == 1
        # Initialize res_init to compute residual norm
        V = VectorOfVectors(size(Ais[1],1), 0, T)
        res_init = residual_norm_affine_init(Ap, bp, V)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v)
    else
        # Initialize res_init to compute residual norm
        V = VectorOfVectors(size(Ais[1],1), 0, T)
        res_init = residual_norm_affine_proj_init(Ap, bp, V)
        compute_res_norm = (u_r, p) -> residual_norm_affine_online(res_init,u_r,p)
        update_res_norm! = (v) -> add_col_to_V!(res_init, v)
    end
    # Form greedy method object
    greedy_method = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ap, bp, ps_greedy, V1, 
                                            VtAVp1, Vtbp1, A_alloc, b_alloc, ualloc1,
                                            Float64[], Float64[])

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
    _, _, _, V2, VtAVp2, Vtbp2, ps_greedy2 = init_affine_rbm(Ap, bp, T)
    strong_greedy = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ap, bp, ps_greedy2, V2, 
                                            VtAVp2, Vtbp2, A_alloc, b_alloc, ualloc1,
                                            Float64[], Float64[])
    # POD/PCA
    pca_proj = pca_projector(truth_sols_mat, num_snapshots)
    ret_data[:pca_projector] = pca_proj
    _, _, _, V3, VtAVp3, Vtbp3, ps_greedy3 = init_affine_rbm(Ap, bp, T)
    pca_method = Greedy_RB_Affine_Linear(approx_stability_factor, compute_res_norm,
                                            update_res_norm!, Ap, bp, ps_greedy3, V3, 
                                            VtAVp3, Vtbp3, A_alloc, b_alloc, ualloc1,
                                            Float64[], Float64[])
    
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

# Deprecated for call with APArray objects
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
    Ap = APArray(Ais, makeθAi)
    bp = APArray(bis, makeθbi)
    return greedy_rb_err_data(param_disc, Ap, bp, approx_stability_factor, num_snapshots, 
                              res_calc=res_calc, T=T, noise=noise)
end