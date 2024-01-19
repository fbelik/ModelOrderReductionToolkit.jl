using LinearAlgebra
using JuMP
using Tulip
using NearestNeighbors
using StaticArrays
using Printf

const LP_MAX_VAL = 1e6

"""
`SCM_Init` is a struct for holding all of the variables
and methods necessary for running the successive constraint
method on an affinely-parameter-dependent matrix
`A(p) = ∑_{i=1}^QA θ_i(p) A_i` to compute a lower-bound
approximation to the minimum singular value of `A`.
"""
struct SCM_Init
    d::Int
    QA::Int
    tree::NNTree
    tree_lookup::AbstractVector{Int}
    B::AbstractVector{Tuple{Float64,Float64}}
    C::AbstractVector{<:AbstractVector{<:Real}}
    C_indices::Vector{Int}
    Mα::Int
    Mp::Int
    ϵ::AbstractFloat
    J::Function
    make_UB::Function
    Y_UB::AbstractVector{<:AbstractVector{<:Real}}
    σ_UBs::AbstractVector{Float64}
    spd::Bool
    R::Float64
    optimizer::Any
end

function Base.show(io::Core.IO, scm::SCM_Init)
    res = "Initialized SCM Object for parametrized"
    if scm.spd
        res *= " symmetric\npositive definite "
    else
        res *= "\n"
    end
    res *= "matrix A(p) = ∑_{i=1}^QA θ_i(p) A_i.\n"
    res *= @sprintf("\nMα=%d, Mp=%d, ϵ=%.4e.",scm.Mα,scm.Mp,scm.ϵ)

    print(io, res)
end

"""
`initialize_SCM_SPD(param_disc,Ais,makeθAi,Mα,Mp,ϵ;[optimizer=Tulip.Optimizer,noise=1]) = SCM_Init`

Method to initialize an `SCM_Init` object to perform the SCM
on an affinely-parameter-dependent symmetric positive definite matrix
`A(p) = ∑_{i=1}^QA θ_i(p) A_i` to compute a lower-bound
approximation to the minimum singular value (or eigenvalue) of `A`.

Parameters:

`param_disc`: Either a matrix where each column is a parameter
value in the discretization, or a vector of parameter vectors.

`Ais`: A vector of matrices, of the same dimension, used to construct
the full matrix `A(p) = ∑_{i=1}^QA makeθAi(p,i) Ais[i]`

`makeθAi(p,i)`: A function that takes in as input a parameter vector
and an index such that `A(p) = ∑_{i=1}^QA makeθAi(p,i) Ais[i]`

`Mα`: Stability constraint constant (a positive integer)

`Mp`: Positivity constraint constant (a positive integer)

`ϵ`: Relative difference allowed between upper-bound and lower-bound approximation
on the parameter discretization (between 0 and 1)

`optimizer`: Optimizer to pass into JuMP Model method for solving
linear programs

`noise`: Determines amount of printed information, between 0 and 2 with 0 being nothing
displayed; default 1
"""
function initialize_SCM_SPD(param_disc::Union{Matrix,Vector},
                        Ais::AbstractVector,
                        makeθAi::Function,
                        Mα::Int,
                        Mp::Int,
                        ϵ::AbstractFloat;
                        optimizer=Tulip.Optimizer,
                        noise::Int=1)
    # Form data tree to search for nearest neighbors
    if typeof(param_disc) <: Vector
        param_disc = reduce(hcat, param_disc)
    end
    tree = KDTree(param_disc)
    tree_lookup = zeros(Int, length(tree.data))
    for i in eachindex(tree.indices)
        tree_lookup[tree.indices[i]] = i 
    end
    tree_lookup = SVector{length(tree.data)}(tree_lookup)
    d = length(tree.data[1])
    # Form boundary set by solving eigenvalue problems on each A_i
    QA = length(Ais)
    # Real numerical range of A equals real numerical range of (1/2)(A+A^T)
    B = Tuple{Float64,Float64}[]
    for i in 1:QA
        A = collect(Ais[i])
        A .= 0.5 .* (A .+ transpose(A))
        eigenvalues = eigen(A).values
        push!(B, (minimum(eigenvalues), maximum(eigenvalues)))
    end
    # Linear program resizing constant
    R = 0.0
    for i in 1:QA
        R = max(R, abs(B[i][1]), abs(B[i][2]))
    end
    R = max(1.0, R / LP_MAX_VAL)
    # Define target functions
    J(p,y) = begin
        res = 0.0
        for i in 1:QA
            res += makeθAi(p,i) * y[i]
        end
        res
    end
    JH(x) = begin
        y = Vector{Float64}(undef,QA)
        for i in 1:QA
            y[i] = (x' * Ais[i] * x) / (x' * x)
        end
        return SVector{QA}(y)
    end
    # Initialize upperbound sets
    p1 = tree.data[1]
    C = [p1]
    C_indices = [1]
    make_UB(p) = begin
        A = zeros(size(Ais[1]))
        for i in eachindex(Ais)
            A .+= makeθAi(p,i) .* Ais[i]
        end
        eg = eigen(A)
        vec = eg.vectors[:,1]
        y = JH(vec)
        σ = eg.values[1]
        return (σ, y)
    end
    σ, y = make_UB(p1)
    Y_UB = [y]
    σ_UBs = [σ]
    scm =  SCM_Init(d, QA, tree, tree_lookup, B, C, 
                    C_indices, Mα, Mp, ϵ, J, make_UB, 
                    Y_UB, σ_UBs, true, R, optimizer)
    # Form upperbound set to ϵ accuracy
    form_upperbound_set!(scm, noise=noise)
    return scm
end

"""
`initialize_SCM_Noncoercive(param_disc,Ais,makeθAi,Mα,Mp,ϵ;[optimizer=Tulip.Optimizer,noise=1]) = SCM_Init`

Method to initialize an `SCM_Init` object to perform the SCM
on an affinely-parameter-dependent matrix
`A(p) = ∑_{i=1}^QA θ_i(p) A_i` to compute a lower-bound
approximation to the minimum singular value of `A`.

Parameters:

`param_disc`: Either a matrix where each column is a parameter
value in the discretization, or a vector of parameter vectors.

`Ais`: A vector of matrices, of the same dimension, used to construct
the full matrix `A(p) = ∑_{i=1}^QA makeθAi(p,i) Ais[i]`

`makeθAi(p,i)`: A function that takes in as input a parameter vector
and an index such that `A(p) = ∑_{i=1}^QA makeθAi(p,i) Ais[i]`

`Mα`: Stability constraint constant (a positive integer)

`Mp`: Positivity constraint constant (a positive integer)

`ϵ`: Relative difference allowed between upper-bound and lower-bound approximation
on the parameter discretization (between 0 and 1)

`optimizer`: Optimizer to pass into JuMP Model method for solving
linear programs

`noise`: Determines amount of printed information, between 0 and 2 with 0 being nothing
displayed; default 1
"""
function initialize_SCM_Noncoercive(param_disc::Union{Matrix,Vector},
                        Ais::AbstractVector{<:AbstractMatrix},
                        makeθAi::Function,
                        Mα::Int,
                        Mp::Int,
                        ϵ::AbstractFloat;
                        optimizer=Tulip.Optimizer,
                        noise::Int=1)
    # Form data tree to search for nearest neighbors
    if typeof(param_disc) <: Vector
        param_disc = reduce(hcat, param_disc)
    end
    tree = KDTree(param_disc)
    tree_lookup = zeros(Int, length(tree.data))
    for i in eachindex(tree.indices)
        tree_lookup[tree.indices[i]] = i 
    end
    tree_lookup = SVector{length(tree.data)}(tree_lookup)
    d = length(tree.data[1])
    # Form boundary set by solving eigenvalue problems on each A_i^T A_j
    QA = length(Ais)
    AiAjs = Matrix{Float64}[]
    for i in 1:QA
        for j in i:QA
            push!(AiAjs, transpose(Ais[i]) * Ais[j])
        end
    end
    QAA = length(AiAjs)
    B = Tuple{Float64,Float64}[]
    # Real numerical range of A equals real numerical range of (1/2)(A+A^T)
    for i in 1:QAA
        AA = collect(AiAjs[i])
        AA .= 0.5 .* (AA .+ transpose(AA))
        eigenvalues = eigen(AA).values
        push!(B, (minimum(eigenvalues), maximum(eigenvalues)))
    end
    # Linear program resizing constant
    R = 0.0
    for i in 1:QAA
        R = max(R, abs(B[i][1]), abs(B[i][2]))
    end
    R = max(1.0, R / LP_MAX_VAL)
    # Define target functions
    J(p,y) = begin
        res = 0.0
        idx = 1
        for i in 1:QA
            θAi = makeθAi(p,i)
            res += θAi^2 * y[idx]
            idx += 1
            for j in (i+1):QA
                res += 2 * θAi * makeθAi(p,j) * y[idx]
                idx += 1
            end
        end
        return res
    end
    JH(x) = begin
        y = Vector{Float64}(undef,QAA)
        for idx in 1:QAA
            y[idx] = (x' * AiAjs[idx] * x) / (x' * x)
        end
        return SVector{QAA}(y)
    end
    # Initialize upperbound sets
    p1 = tree.data[1]
    C = [p1]
    C_indices = [1]
    make_UB(p) = begin
        A = zeros(size(Ais[1]))
        for i in eachindex(Ais)
            A .+= makeθAi(p,i) .* Ais[i]
        end
        svdA = svd(A)
        σ = svdA.S[end]
        vec = svdA.V[:,end]
        y = JH(vec)
        return (σ ^ 2, y)
    end
    σ, y = make_UB(p1)
    Y_UB = [y]
    σ_UBs = [σ]
    scm =  SCM_Init(d, QAA, tree, tree_lookup, B, C, 
                    C_indices, Mα, Mp, ϵ, J, make_UB, 
                    Y_UB, σ_UBs, false, R, optimizer)
    # Form upperbound set to ϵ accuracy
    form_upperbound_set!(scm, noise=noise)
    return scm
end

"""
`solve_LBs_LP(scm_init, p) = (σ_LB, y_LB)`

Helper method that takes in an `SCM_Init_SPD` object and a parameter vector
`p`, and sets up and solves a linear program to compute a lower-bound
`σ_LB` to the minimum singular value of `A(p)` along with the associated vector `y_LB`.
"""
function solve_LBs_LP(scm_init::SCM_Init, p::AbstractVector{<:Real})
    model = Model(scm_init.optimizer)
    try
        set_attribute(model, "IPM_IterationsLimit", 1000000)
        set_attribute(model, "IPM_TimeLimit", 20.0)
    catch e
    end
    set_silent(model)
    @variable(model, y[1:scm_init.QA])
    # Bound constraints
    for i in 1:scm_init.QA
        @constraint(model, scm_init.B[i][1] / scm_init.R <= 
                           y[i] <= 
                           scm_init.B[i][2] / scm_init.R)
    end
    # Stability Constraints
    C_NN_idxs = partialsortperm([norm(p .- p_c) for p_c in scm_init.C], 1:min(length(scm_init.C),scm_init.Mα))
    for i in C_NN_idxs
        p_c = scm_init.C[i]
        ub = scm_init.σ_UBs[i]
        @constraint(model, scm_init.J(p_c,y) >= ub / scm_init.R)
    end
    # Positivity Constraints
    p_idxs, _ = knn(scm_init.tree, p, scm_init.Mp, false, i -> (i in scm_init.C_indices[C_NN_idxs]))
    for p_idx in p_idxs
        p_nn = scm_init.tree.data[scm_init.tree_lookup[p_idx]]
        @constraint(model, scm_init.J(p_nn,y) >= 0)
    end
    @constraint(model, scm_init.J(p,y) >= 0)
    # Minimizer objective
    @objective(model, Min, scm_init.J(p,y))
    optimize!(model)
    if termination_status(model) != OPTIMAL
        println("Warning: Linear Program Solution not optimal, possibly choose different optimizer")
        σ_LB = 0.0
    else
        σ_LB = objective_value(model) * scm_init.R
    end
    y_LB = [value(y[i]) for i in 1:scm_init.QA] .* scm_init.R
    return (σ_LB, y_LB)
end

"""
`form_upperbound_set!(scm_init; [noise=1])`

Helper method that takes in an `SCM_Init_SPD` object and forms the upper-bound
sets `C`, `Y_UB`, and `σ_UBs` until the `ϵ` tolerance provided in `scm_init`
is met across the parameter discretization.  The `noise` input determines the 
amount of printed information, between 0 and 2, with 0 being nothing
displayed, and 1 is default.
"""
function form_upperbound_set!(scm_init::SCM_Init; noise::Int=1)
    while length(scm_init.C) < length(scm_init.tree.data)
        ϵ_k = 0
        σ_UB_k = 0
        σ_LB_k = 0
        p_k = zeros(scm_init.d)
        i_k = 0
        # Loop through every point in discretization to find 
        # arg max {p in discretization} (σ_UB(p) - σ_LB(p)) / σ_UB(p)
        for (i,p_disc) in enumerate(scm_init.tree.data)
            if p_disc in scm_init.C
                continue
            end
            ## Solve linear program to find σ_LB, y_LB
            σ_LB, y_LB = solve_LBs_LP(scm_init, p_disc)
            ## Loop through Y_{UB} to find σ_UB
            σ_UB = Inf
            y_UB = zeros(scm_init.QA)
            for y in scm_init.Y_UB
                Jval = scm_init.J(p_disc, y)
                if Jval < σ_UB
                    σ_UB = Jval
                    y_UB .= y
                end
            end
            ## Compute ϵ_k
            if scm_init.spd
                ϵ_disc = (σ_UB - σ_LB) / σ_UB
            else
                σ_LB = max(0.0, σ_LB)
                ϵ_disc = 1 - sqrt(σ_LB) / sqrt(σ_UB)
            end
            if ϵ_disc < 0.5
                if noise >= 2
                    @printf("ϵ_k found to be %.4e\n",ϵ_disc)
                    @printf("p_disc = %s\n",p_disc)
                    if scm_init.spd
                        @printf("UBs are %.6f, %s\n",σ_UB,y_UB)
                        @printf("LBs are %.6f, %s\n",σ_LB,y_LB)
                    else
                        @printf("UBs are %.6f, %s\n",sqrt(σ_UB),y_UB)
                        @printf("LBs are %.6f, %s\n",sqrt(σ_LB),y_LB)
                    end
                end
            end
            if ϵ_disc > ϵ_k
                if noise >= 2
                    @printf("Updating ϵ_k to %.4e\n",ϵ_disc)
                    @printf("p_disc = %s\n",p_disc)
                    if scm_init.spd
                        @printf("UBs are %.6f, %s\n",σ_UB,y_UB)
                        @printf("LBs are %.6f, %s\n",σ_LB,y_LB)
                    else
                        @printf("UBs are %.6f, %s\n",sqrt(σ_UB),y_UB)
                        @printf("LBs are %.6f, %s\n",sqrt(σ_LB),y_LB)
                    end
                end 
                ϵ_k = ϵ_disc
                p_k = p_disc
                i_k = i
                σ_UB_k = σ_UB
                σ_LB_k = σ_LB
            end
        end
        if noise >= 1
            @printf("k = %d, ϵ_k = %.4e, σ_UB_k = %.4f, σ_LB_k = %.4f, p_k = %s\n", 
                    length(scm_init.C), ϵ_k, scm_init.spd ? σ_UB_k : sqrt(σ_UB_k), scm_init.spd ? σ_LB_k : sqrt(σ_LB_k), p_k)
        end
        if ϵ_k < scm_init.ϵ
            if noise >= 1
                @printf("Terminating on iteration k = %d with ϵ_k=%.4e\n",
                        length(scm_init.C),ϵ_k)
            end
            return
        end
        # Update for next loop
        push!(scm_init.C, p_k)
        push!(scm_init.C_indices, scm_init.tree.indices[i_k])
        σ_k, y_k = scm_init.make_UB(p_k)
        push!(scm_init.Y_UB, y_k)
        push!(scm_init.σ_UBs, σ_k)
    end
    if noise >= 1
        println("Warning: Looped through all of parameter discretization without meeting ϵ bound")
    end
end

"""
`find_sigma_bounds(scm_init, p, [sigma_eps=1.0])`

Method that performs the online phase of SCM for the matrix
`A(p) = ∑_{i=1}^QA θ_i(p) A_i` to compute lower and upper-bound
approximations to the minimum singular value of `A`. Additional
optional parameter `sigma_eps` such that if the computed
ϵ difference of `(σ_UB - σ_LB) / σ_UB` is less than `sigma_eps`, 
we know that not enough stability constraints were enforced, and 
the minimum singular value is directly computed, appended to the 
scm_init's upper-bound set, and returned as both the lower and upper-bounds.
"""
function find_sigma_bounds(scm_init::SCM_Init, p::AbstractVector{<:Real}, 
                           sigma_eps::Float64=1.0)
    # Find lower bound through linear program
    σ_LB, y_LB = solve_LBs_LP(scm_init, p)
    if !scm_init.spd
        σ_LB = sqrt(max(0.0,σ_LB))
    end
    # Loop through Y_{UB} to find upper-bound
    σ_UB = Inf
    for y in scm_init.Y_UB
        Jval = scm_init.J(p, y)
        if Jval < σ_UB
            σ_UB = Jval
        end
    end
    if !scm_init.spd
        σ_UB = sqrt(σ_UB)
    end
    ϵ = (σ_UB - σ_LB) / σ_UB
    if ϵ > sigma_eps
        @printf("Warning: Computed ϵ of %.4e was greater than the tolerance of %.4e.\n", ϵ, sigma_eps)
        println("Recomputing explicitly, adding to upper-bound set, and returning explicit result.")
        push!(scm_init.C, p)
        C_idx = -1
        for (i,ptree) in enumerate(scm_init.tree.data)
            if ptree == p
                C_idx = scm_init.tree.indices[i]
                break
            end
        end
        push!(scm_init.C_indices, C_idx)
        σ, y = scm_init.make_UB(p)
        push!(scm_init.Y_UB, y)
        push!(scm_init.σ_UBs, σ)
        if !scm_init.spd
            σ = sqrt(σ)
        end
        return (σ,σ)
    end
    return (σ_LB, σ_UB)
end