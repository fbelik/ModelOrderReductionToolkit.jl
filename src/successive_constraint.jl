const MAX_ITER = 10000
const LP_MAX_VAL = 1e6

"""
`SCM_Init` is a struct for holding all of the variables
and methods necessary for running the successive constraint
method on an affinely-parameter-dependent matrix
`A(p) = ∑_{i=1}^QA θ_i(p) A_i` to compute a lower-bound
approximation to the minimum singular value of `A`.

It is additionally a functor as calling `scm_init(p)` on
a parameter vector `p` will return the lower-bound estimate
of the minimum singular value of `A(p)`.
"""
struct SCM_Init <: Function
    d::Int
    QA::Int
    tree::NNTree
    tree_lookup::AbstractVector{Int}
    B::AbstractVector{Tuple{Float64,Float64}}
    C::AbstractVector{<:AbstractVector}
    C_indices::Vector{Int}
    Mα::Int
    Mp::Int
    ϵ::AbstractFloat
    J::Function
    make_UB::Function
    Y_UB::AbstractVector{<:AbstractVector{<:Number}}
    σ_UBs::AbstractVector{Float64}
    σ_LBs::AbstractVector{Float64}
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
`(scm_init::SCM_Init)(p::AbstractVector; noise=0)`

Method that performs the online phase of SCM for the matrix
`A(p) = ∑ makeθAi(p,i) Ais[i]` to compute a lower-bound
approximation to the minimum singular value of `A`.
"""
function (scm_init::SCM_Init)(p::AbstractVector; noise=0)
    # Find lower bound through linear program
    σ_LB, _ = solve_LBs_LP(scm_init, p, noise=noise)
    # In case of round error
    σ_LB = max(0.0,σ_LB)
    if !scm_init.spd
        σ_LB = sqrt(σ_LB)
    end
    return σ_LB
end

"""
`initialize_SCM_SPD(param_disc,Ais,makeθAi,Mα,Mp,ϵ;[T=Float64,optimizer=Tulip.Optimizer,kmaxiter=1000,noise=1]) = SCM_Init`

Method to initialize an `SCM_Init` object to perform the SCM
on an affinely-parameter-dependent symmetric positive definite matrix
`A(p) = ∑ makeθAi(p,i) Ais[i]` to compute a lower-bound
approximation to the minimum singular value (or eigenvalue) of `A(p)`.

If wish to use complex matrices, pass in `T=ComplexF64`, however,
makeθAi must be real.

Parameters:

`param_disc`: Either a matrix where each column is a parameter
value in the discretization, or a vector of parameter vectors.

`Ais`: A vector of matrices, of the same dimension, used to construct
the full matrix `A(p) = ∑ makeθAi(p,i) Ais[i]`

`makeθAi(p,i)`: A function that takes in as input a parameter vector
and an index such that `A(p) = ∑ makeθAi(p,i) Ais[i]`

`Mα`: Stability constraint constant (a positive integer)

`Mp`: Positivity constraint constant (a positive integer)

`ϵ`: Relative difference allowed between upper-bound and lower-bound approximation
on the parameter discretization (between 0 and 1)

`T`: Datatype for matrix initialization, default `Float64`. If using 
complex matrices, pass `T=ComplexF64`.

`optimizer`: Optimizer to pass into JuMP Model method for solving
linear programs

`kmaxiter`: Maximum number of iterations used in iterating eigensolver before
defaulting to full-dense eigensolve.

`noise`: Determines amount of printed information, between 0 and 2 with 0 being nothing
displayed; default 1
"""
function initialize_SCM_SPD(param_disc::Union{Matrix,Vector},
                            Ais::AbstractVector,
                            makeθAi::Function,
                            Mα::Int,
                            Mp::Int,
                            ϵ::AbstractFloat;
                            T::Type=Float64,
                            optimizer=Tulip.Optimizer,
                            kmaxiter=1000,
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
    Ais_herm = []
    for i in 1:QA
        A = 0.5 .* (Ais[i] .+ Ais[i]')
        push!(Ais_herm, A)
        mineig = smallest_real_eigval(A, kmaxiter, noise)
        maxeig = largest_real_eigval(A, kmaxiter, noise)
        push!(B, (mineig, maxeig))
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
        real(res)
    end
    y_JH = Vector{Float64}(undef,QA)
    JH(x) = begin
        for i in 1:QA
            y_JH[i] = real((x' * Ais_herm[i] * x) / (x' * x))
        end
        return SVector{QA}(y_JH)
    end
    # Initialize upperbound sets
    p1 = tree.data[1]
    C = [p1]
    C_indices = [1]
    A_UB = zeros(T,size(Ais[1]))
    make_UB(p) = begin
        A_UB .= 0
        for i in eachindex(Ais)
            A_UB .+= makeθAi(p,i) .* Ais[i]
        end
        σ, vec = smallest_real_pos_eigpair(A_UB, kmaxiter, noise)
        y = JH(vec)
        return (σ, y)
    end
    σ, y = make_UB(p1)
    Y_UB = [y]
    σ_UBs = [σ]
    σ_LBs = zeros(length(tree.data))
    scm =  SCM_Init(d, QA, tree, tree_lookup, B, C, 
                    C_indices, Mα, Mp, ϵ, J, make_UB, 
                    Y_UB, σ_UBs, σ_LBs, true, R, optimizer)
    # Form upperbound set to ϵ accuracy
    form_upperbound_set!(scm, noise=noise)
    return scm
end

"""
`initialize_SCM_Noncoercive(param_disc,Ais,makeθAi,Mα,Mp,ϵ;[T=Float64,optimizer=Tulip.Optimizer,kmaxiter=1000,noise=1]) = SCM_Init`

Method to initialize an `SCM_Init` object to perform the SCM
on an affinely-parameter-dependent matrix
`A(p) = ∑ makeθAi(p,i) Ais[i]` to compute a lower-bound
approximation to the minimum singular value of `A`.

If wish to use complex matrices, pass in `T=ComplexF64`, however,
makeθAi must be real.

Parameters:

`param_disc`: Either a matrix where each column is a parameter
value in the discretization, or a vector of parameter vectors.

`Ais`: A vector of matrices, of the same dimension, used to construct
the full matrix `A(p) = ∑ makeθAi(p,i) Ais[i]`

`makeθAi(p,i)`: A function that takes in as input a parameter vector
and an index such that `A(p) = ∑ makeθAi(p,i) Ais[i]`

`Mα`: Stability constraint constant (a positive integer)

`Mp`: Positivity constraint constant (a positive integer)

`ϵ`: Relative difference allowed between upper-bound and lower-bound approximation
on the parameter discretization (between 0 and 1)

`T`: Datatype for matrix initialization, default `Float64`. If using 
complex matrices, pass `T=ComplexF64`.

`optimizer`: Optimizer to pass into JuMP Model method for solving
linear programs

`kmaxiter`: Maximum number of iterations used in iterating eigensolver before
defaulting to full-dense eigensolve.

`noise`: Determines amount of printed information, between 0 and 2 with 0 being nothing
displayed; default 1
"""
function initialize_SCM_Noncoercive(param_disc::Union{Matrix,Vector},
                        Ais::AbstractVector,
                        makeθAi::Function,
                        Mα::Int,
                        Mp::Int,
                        ϵ::AbstractFloat;
                        T::Type=Float64,
                        optimizer=Tulip.Optimizer,
                        kmaxiter=1000,
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
    AiAjs_herm = []
    B = Tuple{Float64,Float64}[]
    for i in 1:QA
        for j in i:QA
            AiAj = Ais[i]' * Ais[j]
            AiAj .= 0.5 .* (AiAj .+ AiAj')
            push!(AiAjs_herm, AiAj)
            mineig = smallest_real_eigval(AiAj, kmaxiter, noise)
            maxeig = largest_real_eigval(AiAj, kmaxiter, noise)
            push!(B, (mineig, maxeig))
        end
    end
    QAA = length(AiAjs_herm)
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
                θAj = makeθAi(p,j)
                res += 2 * θAi * θAj * y[idx]
                idx += 1
            end
        end
        return res
    end
    y_JH = Vector{Float64}(undef,QAA)
    JH(x) = begin
        for idx in 1:QAA
            y_JH[idx] = real((x' * AiAjs_herm[idx] * x) / (x' * x))
        end
        return SVector{QAA}(y_JH)
    end
    # Initialize upperbound sets
    p1 = tree.data[1]
    C = [p1]
    C_indices = [1]
    A_UB = zeros(T,size(Ais[1]))
    make_UB(p) = begin
        A_UB .= 0
        for i in eachindex(Ais)
            A_UB .+= makeθAi(p,i) .* Ais[i]
        end
        # Compute minimum eigenvalue of A'A
        A_UB .= A_UB' * A_UB
        σ², vec = smallest_real_pos_eigpair(A_UB, kmaxiter, noise)
        y = JH(vec)
        return (σ², y)
    end
    σ², y = make_UB(p1)
    Y_UB = [y]
    σ_UBs = [σ²]
    σ_LBs = zeros(length(tree.data))
    scm =  SCM_Init(d, QAA, tree, tree_lookup, B, C, 
                    C_indices, Mα, Mp, ϵ, J, make_UB, 
                    Y_UB, σ_UBs, σ_LBs, false, R, optimizer)
    # Form upperbound set to ϵ accuracy
    form_upperbound_set!(scm, noise=noise)
    return scm
end

"""
`solve_LBs_LP(scm_init, p[; noise=1]) = (σ_LB, y_LB)`

Helper method that takes in an `SCM_Init_SPD` object and a parameter vector
`p`, and sets up and solves a linear program to compute a lower-bound
`σ_LB` to the minimum singular value of `A(p)` along with the associated vector `y_LB`.
"""
function solve_LBs_LP(scm_init::SCM_Init, p::AbstractVector; noise=1)
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
    p_idxs, _ = knn(scm_init.tree, p, scm_init.Mp, true, i -> (i in scm_init.C_indices[C_NN_idxs]))
    tree_idx = scm_init.tree_lookup[p_idxs[1]]
    closest_p = scm_init.tree.data[tree_idx]
    if scm_init.Mp >= 1 && closest_p == p
        @constraint(model, scm_init.J(p,y) >= scm_init.σ_LBs[tree_idx] / scm_init.R)
    else
        @constraint(model, scm_init.J(p,y) >= 0)
        @constraint(model, scm_init.J(closest_p,y) >= 0)
    end
    for p_idx in p_idxs[2:end]
        tree_idx = scm_init.tree_lookup[p_idx]
        p_nn = scm_init.tree.data[tree_idx]
        @constraint(model, scm_init.J(p_nn,y) >= 0)
    end
    # Minimizer objective
    @objective(model, Min, scm_init.J(p,y))
    optimize!(model)
    if termination_status(model) != OPTIMAL
        if noise >= 1
            println("Warning: Linear Program Solution not optimal, possibly choose different optimizer")
        end
        σ_LB = 0.0
        y_LB = zeros(scm_init.QA)
    elseif objective_value(model) < 0.0
        if noise >= 2
            println("Warning: Lower bound found to be negative")
        end
        σ_LB = 0.0
        y_LB = zeros(scm_init.QA)
    else
        σ_LB = objective_value(model) * scm_init.R
        y_LB = [value(y[i]) for i in 1:scm_init.QA] .* scm_init.R
    end
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
    if noise >= 1
        println("Beginning SCM Procedure")
        println("----------")
    end
    while length(scm_init.C) < length(scm_init.tree.data)
        ϵ_k = -1.0
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
            σ_LB, y_LB = solve_LBs_LP(scm_init, p_disc, noise=noise)
            if σ_LB >= scm_init.σ_LBs[i]
                scm_init.σ_LBs[i] = σ_LB
            end
            ## Loop through Y_{UB} to find σ_UB
            σ_UB = Inf
            y_UB = zeros(eltype(scm_init.Y_UB[1]),scm_init.QA)
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
                σ_UB = max(0.0, σ_UB)
                ϵ_disc = 1 - sqrt(σ_LB) / sqrt(σ_UB)
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
                println("----------")
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
`find_sigma_bounds(scm_init, p[, sigma_eps=1.0; noise=0])`

Method that performs the online phase of SCM for the matrix
`A(p) = ∑ makeθAi(p,i) Ais[i]` to compute lower and upper-bound
approximations to the minimum singular value of `A`. Additional
optional parameter `sigma_eps` such that if the computed
ϵ difference of `(σ_UB - σ_LB) / σ_UB` is less than `sigma_eps`, 
we know that not enough stability constraints were enforced, and 
the minimum singular value is directly computed, appended to the 
scm_init's upper-bound set, and returned as both the lower and upper-bounds.
"""
function find_sigma_bounds(scm_init::SCM_Init, p::AbstractVector, 
                           sigma_eps::Float64=1.0; noise=0)
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
    # Find lower bound through linear program
    σ_LB, y_LB = solve_LBs_LP(scm_init, p, noise=noise)
    if !scm_init.spd
        σ_LB = sqrt(max(0.0,σ_LB))
    end
    # In case of roundoff error
    σ_LB = min(σ_LB, σ_UB)
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