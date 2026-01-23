const LP_MAX_RESCALE = 1e6

abstract type AbstractSCM <: Function end

function Base.show(io::Core.IO, scm::AbstractSCM)
    res = "SCM Object for matrix "
    print(io, res)
    print(io, scm.Ap)
end

"""
`SCM{P} <: AbstractSCM <: Function where P <: Int`

An SCM object for a SPD APArray matrix `Ap`. For non-SPD 
matrix `Ap`, can form Grammian matrix `Ap'Ap` and perform
SCM on it.

`scm = SCM(Ap::APArray, ps::AbstractVector[, ϵ=0.8, Mα=20, Mp=0; coercive=false, 
optimizer=HiGHS.Optimizer, lp_attrs=Dict(), make_monotonic=true, max_iter=500, noise=0])`

Constructs an `SCM` object for `Ap` about parameter values `ps`. If `coercive=false`,
then constructs the Grammian matrix `Ap(p)'Ap(p)`. If `hermitianpart(Ap(p))` is positive definite
for all `p`, then set `coercive=true`. Constrains the `SCM` object to a relative gap of `ϵ`. If 
set `ϵ=nothing`, then does not perform constraining. `Mα` and `Mp` are the stability and 
positivity parameters, `optimizer` is the optimizer used for solving LPs in JuMP. `lp_attrs`
can contain additional attributes to pass into the JuMP model (optimizer dependent). 
`make_monotonic` ensures that the lower-bound predictions increase monotonically. `max_iter` 
is the maximum number of SCM constraining iterations. `noise` determines the amount of printed output.

The `scm` object may be used by calling it on a parameter vector

`scm(p[, Mα=20, Mp=0; which=:L, noise=0])`

Which returns the following:

- `which=:L` - The lower-bound prediction at `p`
- `which=:U` - The upper-bound prediction at `p`
- `which=:E` - The relative gap `1 - LB/UB` at `p`
"""
mutable struct SCM{P} <: AbstractSCM where P <: Int
    const Ap::APArray
    const B::Vector{Tuple{Float64,Float64}}
    const UBs::Dict{SVector{P,Float64},Tuple{Float64,Vector{Float64}}}
    const σ_LBs::Dict{SVector{P,Float64},Float64}
    const coercive::Bool
    const lp_model::JuMP.Model
    const optimizer
    const R::Float64
    kdtree::KDTree
    pos_kdtree::KDTree
end

"""
`ANLSCM{P} <: AbstractSCM <: Function where P <: Int`

WARNING: Below method is experimental and still in development.

An SCM object for an APArray matrix `Ap` which uses nonlinear
constraining. 

`scm = ANLSCM(Ap::APArray, ps::AbstractVector[, ϵ=0.8, Mα=20; optimizer=Ipopt.Optimizer, 
lp_attrs=Dict(), nonlin_alpha=0.0, max_iter=500, noise=0, constrain_kwargs...])`

Constructs an `ANLSCM` object for `Ap` about parameter values `ps`. Constrains the `ANLSCM` 
object to a relative gap of `ϵ`. If set `ϵ=nothing`, then does not perform constraining. 
`Mα` is the stability parameter. `optimizer` is the optimizer used for solving NLPs in JuMP. `lp_attrs`
can contain additional attributes to pass into the JuMP model (optimizer dependent). 
`nonlin_alpha` is the initial value of nonlinear constraint coefficients (0 for lightest constraint, 1 for tightest and LB guarantee).
`max_iter` is the maximum number of SCM constraining iterations. `noise` determines the amount of printed output.
Additional kwargs passed to `constrain!`.

The `scm` object may be used by calling it on a parameter vector

`scm(p[, Mα=20; which=:L, noise=0])`

Which returns the following:

- `which=:L` - The lower-bound prediction at `p`
- `which=:U` - The upper-bound prediction at `p`
- `which=:E` - The relative gap `1 - LB/UB` at `p`
"""
mutable struct ANLSCM{P} <: AbstractSCM where P <: Int
    const Ap::APArray
    const B::Vector{Tuple{Float64,Float64}}
    const UBs::Dict{SVector{P,Float64},Tuple{Float64,Vector{Float64}}}
    const σ_LBs::Dict{SVector{P,Float64},Float64}
    const nlp_model::JuMP.Model
    const optimizer
    const R::Float64
    kdtree::KDTree
end

"""
`NNSCM{P} <: AbstractSCM <: Function where P <: Int`

An SCM object for an APArray matrix `Ap` which uses domain decomposition
and works in a natural norm. See Huynh et al., 2010, A natural norm...

`scm = NNSCM(Ap::APArray, ps::AbstractVector[, ϵ=0.8, ϵ_β=0.8, ϕ=0.0, Mα=20; optimizer=HiGHS.Optimizer, 
lp_attrs=Dict(), max_iter=500, max_inner_iter=500, noise=0, constrain_kwargs...])`

Constructs an `NNSCM` object for `Ap` about parameter values `ps`. Constrains the `NNSCM` 
object to a relative gap of `ϵ`. If set `ϵ=nothing`, then does not perform constraining. 
`ϵ_β` is the natural-norm relative gap ensured within subdomains.
`ϕ` is the minimum natural-norm lower-bound prediction required to be considered within a domain.
`Mα` is the stability parameter. `optimizer` is the optimizer used for solving NLPs in JuMP. `lp_attrs`
can contain additional attributes to pass into the JuMP model (optimizer dependent). 
`max_iter` is the maximum number of SCM constraining iterations and `max_inner_iter` is the
maximum number of iterations within a particular domain decomposition. 
`noise` determines the amount of printed output.
Additional kwargs passed to `constrain!`.

The `scm` object may be used by calling it on a parameter vector

`scm(p[, Mα=20; pbar=nothing, which=:L, noise=0])`

Which returns the following:

- `which=:L` - The lower-bound prediction at `p`
- `which=:U` - The upper-bound prediction at `p`
- `which=:E` - The relative gap `1 - LB/UB` at `p`

If `pbar` is not nothing, then if `pbar in keys(scm.UBs)`, solves the above restricted
to the domain decomposition about `pbar`.
"""
struct NNSCM{P} <: AbstractSCM where P <: Int
    Ap::APArray
    AAp::APArray
    σ_maxs::Vector{Float64}
    β_UBs::Dict{SVector{P,Float64},Dict{SVector{P,Float64},Tuple{Float64,Vector{Float64}}}}
    UBs::Dict{SVector{P,Float64},Tuple{Float64,Vector{Float64}}}
    β_LBs::Dict{SVector{P,Float64},Dict{SVector{P,Float64},Float64}}
    σ_LBs::Dict{SVector{P,Float64},Float64}
    lp_model::JuMP.Model
    optimizer
    R::Float64
    kdtrees::Dict{SVector{P,Float64},KDTree}
end

function form_SPD_bounding_boxes(Ap_herm::APArray; noise=0, reigkwargs...)
    B = Tuple{Float64,Float64}[]
    for (d,A) in enumerate(Ap_herm.arrays)
        if noise >= 1
            println("Forming bounding box $d/$(length(Ap_herm.arrays))")
        end
        mineig,_ = reig(A, which=:S, noise=noise; reigkwargs...)
        maxeig,_ = reig(A, which=:L, noise=noise; reigkwargs...)
        push!(B, (mineig, maxeig))    
    end
    
    if noise >= 1
        println("Finished forming bounding boxes")
    end
    return B
end

function compute_max_svals(Ap_herm::APArray; noise=0, reigkwargs...)
    σ_maxs = Vector{Float64}(undef, length(Ap_herm.arrays))
    for (d,A) in enumerate(Ap_herm.arrays)
        if noise >= 1
            println("Computing maximal singular value $d/$(length(Ap_herm.arrays))")
        end
        σ_maxs[d] = largest_sval(A, noise=noise, kmaxiter=get(reigkwargs,:kmaxiter,1000))
    end
    if noise >= 1
        println("Finished computing maximal singular values")
    end
    return σ_maxs
end

function make_JuMP_model(optimizer, QA::Int, B::Union{Nothing,Vector{Tuple{Float64,Float64}}}, R, lp_attrs=Dict(); nonlinear=false, nonlin_alpha=0.0, noise=0)
    model = JuMP.Model(optimizer)
    for (attr,value) in  lp_attrs
        try
            set_attribute(model, attr, value)
        catch
            if noise >= 1
                println("The optimizer $(optimizer) does not have an attribute $(attr) to set")
            end
        end
    end
    set_silent(model)
    @variable(model, y[1:QA])
    if !isnothing(B)
        for i in 1:QA
            @constraint(model, B[i][1] / R <= y[i] <= B[i][2] / R)
        end
    end
    if nonlinear
        idx = 1
        QA = Int((-1 + sqrt(1 + 8 * QA)) / 2)
        for i in 1:QA
            idx += 1
            for j in (i+1):QA
                idxii = idx - (j - i)
                idxjj = idx + (QA - j + 1)
                for jct in i+1:(j-1)
                    idxjj += (QA - jct + 1)
                end
                init_α = 0.0
                if !isnothing(B)
                    if B[idx][1] > 0 && B[idxii][2] > 0 && B[idxjj][2] > 0
                        init_α = B[idx][1] ^ 2 / (B[idxii][2] * B[idxjj][2] - eps())
                    elseif B[idx][2] < 0 && B[idxii][2] > 0 && B[idxjj][2] > 0
                        init_α = B[idx][2] ^ 2 / (B[idxii][2] * B[idxjj][2] - eps())
                    end
                end
                init_α = min(1.0, max(init_α, nonlin_alpha))
                @constraint(model, y[idx]*y[idx] <= init_α * y[idxii] * y[idxjj], base_name="$i-$j")
                idx += 1
            end
        end
    end
    return model
end

function tohermitian(A::AbstractMatrix)
    return 0.5 .* (A .+ A')
end

function SCM(Ap::APArray, ps::AbstractVector, ϵ::Union{Real,Nothing}=0.8, Mα::Int=20, Mp::Int=0; 
                 coercive=false, optimizer=HiGHS.Optimizer, lp_attrs=Dict(), make_monotonic=true, max_iter=500, noise=0)
    Q, Ap_herm = begin
        if coercive
            length(Ap.arrays), APArray(tohermitian.(Ap.arrays), Ap.makeθi, Ap.precompθ)
        else
            QA = length(Ap.arrays)
            QAA = Int(QA * (QA + 1) / 2)
            inds = [(i,j) for i in 1:QA for j in i:QA]
            AiAjs_herm = [tohermitian(Ap.arrays[i]' * Ap.arrays[j]) for (i,j) in inds]
            if Ap.precompθ
                makeθi = (p) -> begin
                    θis = Ap.makeθi(p)
                    res = Vector{Float64}(undef, QAA)
                    for idx in 1:QAA
                        i,j = inds[idx]
                        res[idx] = (2 - (i==j)) * θis[i] * θis[j]
                    end
                    res
                end
            else
                makeθi = (p,idx) -> begin
                    (i,j) = inds[idx]
                    (2 - (i==j)) * Ap.makeθi(p,i) * Ap.makeθi(p,j)
                end
            end
            QAA, APArray(AiAjs_herm, makeθi, Ap.precompθ)
        end
    end
    B = form_SPD_bounding_boxes(Ap_herm, noise=noise)
    R = 1.0
    for i in 1:length(Ap_herm.arrays)
        R = max(R, abs(B[i][1]), abs(B[i][2]))
    end
    R = max(1.0, R / LP_MAX_RESCALE)
    model = make_JuMP_model(optimizer, Q, B, R, lp_attrs, noise=noise)
    P = length(ps[1])
    scm = SCM{P}(Ap_herm, B, Dict(), Dict(), coercive, model, optimizer, R, KDTree(zeros(P,0)), KDTree(zeros(P,0)))
    if !isnothing(ϵ)
        constrain!(scm, ps, ϵ, Mα, Mp, make_monotonic=make_monotonic, max_iter=max_iter, noise=noise)
    end
    return scm
end

function ANLSCM(Ap::APArray, ps::AbstractVector, ϵ::Union{Real,Nothing}=0.8, Mα::Int=20; optimizer=Ipopt.Optimizer, 
                lp_attrs=Dict(), nonlin_alpha=0.0, max_iter=500, noise=0, constrain_kwargs...)
    QA = length(Ap.arrays)
    QAA = Int(QA * (QA + 1) / 2)
    inds = [(i,j) for i in 1:QA for j in i:QA]
    AiAjs_herm = [tohermitian(Ap.arrays[i]' * Ap.arrays[j]) for (i,j) in inds]
    if Ap.precompθ
        makeθi = (p) -> begin
            θis = Ap.makeθi(p)
            res = Vector{Float64}(undef, QAA)
            for idx in 1:QAA
                i,j = inds[idx]
                res[idx] = (2 - (i==j)) * θis[i] * θis[j]
            end
            res
        end
    else
        makeθi = (p,idx) -> begin
            (i,j) = inds[idx]
            (2 - (i==j)) * Ap.makeθi(p,i) * Ap.makeθi(p,j)
        end
    end
    AAp_herm = APArray(AiAjs_herm, makeθi, Ap.precompθ)
    B = form_SPD_bounding_boxes(AAp_herm, noise=noise)
    R = 1.0
    for i in 1:QA
        R = max(R, abs(B[i][1]), abs(B[i][2]))
    end
    R = max(1.0, R / LP_MAX_RESCALE)
    model = make_JuMP_model(optimizer, QAA, B, R, lp_attrs, nonlinear=true, nonlin_alpha=nonlin_alpha, noise=noise)
    P = length(ps[1])
    scm = ANLSCM{P}(AAp_herm, B, Dict(), Dict(), model, optimizer, R, KDTree(zeros(P,0)))
    if !isnothing(ϵ)
        constrain!(scm, ps, ϵ, Mα, max_iter=max_iter, noise=noise; constrain_kwargs...)
    end
    return scm
end

function NNSCM(Ap::APArray, ps::AbstractVector, ϵ::Union{Real,Nothing}=0.8, ϵ_β::Real=0.8, ϕ::Real=0.0, Mα::Int=20; 
               optimizer=HiGHS.Optimizer, lp_attrs=Dict(), max_iter=500, max_inner_iter=500, noise=0, constrain_kwargs...)
    QA = length(Ap.arrays)
    QAA = Int(QA * (QA + 1) / 2)
    inds = [(i,j) for i in 1:QA for j in i:QA]
    AiAjs_herm = [tohermitian(Ap.arrays[i]' * Ap.arrays[j]) for (i,j) in inds]
    if Ap.precompθ
        makeθi = (p) -> begin
            θis = Ap.makeθi(p)
            res = Vector{Float64}(undef, QAA)
            for idx in 1:QAA
                i,j = inds[idx]
                res[idx] = (2 - (i==j)) * θis[i] * θis[j]
            end
            res
        end
    else
        makeθi = (p,idx) -> begin
            (i,j) = inds[idx]
            (2 - (i==j)) * Ap.makeθi(p,i) * Ap.makeθi(p,j)
        end
    end
    AAp_herm = APArray(AiAjs_herm, makeθi, Ap.precompθ)
    σ_maxs = compute_max_svals(Ap, noise=noise)
    R = max(1.0, σ_maxs...)
    R = max(1.0, R / LP_MAX_RESCALE)
    model = make_JuMP_model(optimizer, QA, nothing, R, lp_attrs, noise=noise)
    P = length(ps[1])
    scm = NNSCM{P}(Ap, AAp_herm, σ_maxs, Dict(), Dict(), Dict(), Dict(), model, optimizer, R, Dict())
    if !isnothing(ϵ)
        constrain!(scm, ps, ϵ, ϵ_β, ϕ, Mα, max_iter=max_iter, max_inner_iter=max_inner_iter, noise=noise; constrain_kwargs...)
    end
    return scm
end

"""
`copy_scm(scm::SCM[; lp_attrs=Dict(), noise=0])`

Makes a copy of `scm` and provides `lp_attrs` to the new JuMP model.
"""
function copy_scm(scm::SCM{P}; lp_attrs=Dict(), noise=0) where P
    Q = length(scm.Ap.arrays)
    model = make_JuMP_model(scm.optimizer, Q, scm.B, scm.R, lp_attrs, noise=noise)
    return SCM{P}(scm.Ap, scm.B, copy(scm.UBs), copy(scm.σ_LBs), scm.coercive, model, scm.optimizer, scm.R, deepcopy(scm.kdtree), deepcopy(scm.pos_kdtree))
end

"""
`copy_scm(scm::ANLSCM[; nonlin_alpha=0.0, lp_attrs=Dict(), noise=0])`

Makes a copy of `scm` with initial nonlinear constraint coefficients `nonlin_alpha` 
and provides `lp_attrs` to the new JuMP model.
"""
function copy_scm(scm::ANLSCM{P}; nonlin_alpha=0.0, lp_attrs=Dict(), noise=0)  where P
    Q = length(scm.Ap.arrays)
    model = make_JuMP_model(scm.optimizer, Q, scm.B, scm.R, lp_attrs, nonlinear=true, nonlin_alpha=nonlin_alpha, noise=noise)
    return ANLSCM{P}(scm.Ap, scm.B, deepcopy(scm.UBs), copy(scm.σ_LBs), model, scm.optimizer, scm.R, deepcopy(scm.kdtree))
end

"""
`copy_scm(scm::NNSCM[; lp_attrs=Dict(), noise=0])`

Makes a copy of `scm` and provides `lp_attrs` to the new JuMP model.
"""
function copy_scm(scm::NNSCM{P}; lp_attrs=Dict(), noise=0)  where P
    Q = length(scm.Ap.arrays)
    model = make_JuMP_model(scm.optimizer, Q, nothing, scm.R, lp_attrs, noise=noise)
    return NNSCM{P}(scm.Ap, scm.AAp, scm.σ_maxs, deepcopy(scm.β_UBs), deepcopy(scm.UBs), deepcopy(scm.β_LBs), deepcopy(scm.σ_LBs), model, scm.optimizer, scm.R, deepcopy(scm.kdtrees))
end

function J(scm::AbstractSCM, p, y; Ap=scm.Ap)
    if Ap.precompθ
        return dot(Ap.makeθi(p), y)
    else
        res = 0.0
        for i in eachindex(y)
            res += Ap.makeθi(p,i) * y[i]
        end
        return res
    end
end

function make_σ_UB(scm::AbstractSCM, p; noise=0, reigkwargs...)
    Ap = isa(scm, NNSCM) ? scm.AAp : scm.Ap
    A_UB = Ap(p)
    σ, x = reig(A_UB, which=:SP, noise=noise; reigkwargs...)
    QA = length(Ap.arrays)
    y = Vector{Float64}(undef, QA)
    for i in 1:QA
        y[i] = real(dot(x, Ap.arrays[i], x) / dot(x, x))
    end
    return (σ, y)
end

function make_β_UB(scm::NNSCM, p, pbar; noise=0, reigkwargs...)
    Ap = scm.Ap
    QA = length(Ap.arrays)
    A_pbar = Ap(pbar)
    A_p = (p == pbar) ? A_pbar : Ap(p)
    β, x = reig(tohermitian(A_pbar' * A_p), (A_pbar' * A_pbar), which=:S, noise=noise, force_sigma=1.0; reigkwargs...)
    y = Vector{Float64}(undef, QA)
    A_pbarx = (A_pbar * x)
    denom = dot(A_pbarx, A_pbarx)
    for i in 1:QA
        y[i] = real(dot(x, tohermitian(A_pbar' * Ap.arrays[i]), x) / denom)
    end
    return (β, y)
end

function solve_LB(scm::SCM, p, Mα=20, Mp=0; noise=0)
    model = scm.lp_model
    y = all_variables(model)
    # Remove old stability and positivity constraints
    for (F, S) in list_of_constraint_types(model)  
        if !(S <: MathOptInterface.Interval || F <: JuMP.QuadExpr || S <: MathOptInterface.PowerCone)
            delete(model, all_constraints(model, F, S))
        end
    end
    # Stability Constraints
    if !isnothing(scm.kdtree)
        C_NN_idxs, _ = knn(scm.kdtree, p, min(Mα, length(scm.kdtree.data)))
        for i in C_NN_idxs
            p_c = scm.kdtree.data[i]
            ub = scm.UBs[p_c][1]
            @constraint(model, J(scm,p_c,y) >= ub / scm.R)
        end
    end
    # Lower-bound Constraint 
    if p in keys(scm.σ_LBs)
        @constraint(model, J(scm,p,y) >= scm.σ_LBs[p] / scm.R)
    end
    if Mp > 0 && !isnothing(scm.pos_kdtree)
        # Positivity Constraints
        pos_kdtree = scm.pos_kdtree
        idxs, _ = knn(pos_kdtree, p, min(Mp,length(pos_kdtree.data)))
        for idx in idxs
            tree_idx = findfirst(isequal(idx), pos_kdtree.indices)
            if isnothing(tree_idx)
                continue
            end
            p_p = pos_kdtree.data[tree_idx] 
            if p_p == p && p in keys(scm.σ_LBs)
                continue
            end
            @constraint(model, J(scm,p_p,y) >= 0.0)
        end
    end
    # Objective
    @objective(model, Min, J(scm,p,y))
    # Solve
    optimize!(model)
    # Return
    if !(termination_status(model) in (OPTIMAL, LOCALLY_SOLVED, ALMOST_OPTIMAL))
        if noise >= 1
            println("Warning: Linear Program Solution not optimal; termination status: $(termination_status(model))")
            println("Setting σ_LB=0")
        end
        σ_LB = 0.0
        y_LB = zeros(length(y))
    else
        y_LB = value.(y) .* scm.R
        σ_LB = objective_value(model) * scm.R
    end
    return (σ_LB, y_LB)
end

function solve_LB(scm::ANLSCM, p, Mα=20; noise=0)
    model = scm.nlp_model
    y = all_variables(model)
    # Remove old stability and positivity constraints
    for (F, S) in list_of_constraint_types(model)  
        if !(S <: MathOptInterface.Interval || F <: JuMP.QuadExpr || S <: MathOptInterface.PowerCone)
            delete(model, all_constraints(model, F, S))
        end
    end
    # Stability Constraints
    C_NN_idxs, _ = knn(scm.kdtree, p, min(Mα, length(scm.kdtree.data)))
    for i in C_NN_idxs
        p_c = scm.kdtree.data[i]
        ub = scm.UBs[p_c][1]
        @constraint(model, J(scm,p_c,y) >= ub / scm.R)
    end
    @objective(model, Min, J(scm,p,y))
    optimize!(model)
    if !(termination_status(model) in (OPTIMAL, LOCALLY_SOLVED, ALMOST_OPTIMAL))
        if noise >= 1
            println("Warning: Nonlinear Program Solution not optimal; termination status: $(termination_status(model))")
            println("Setting σ_LB=-Inf")
        end
        σ_LB = -Inf
        y_LB = zeros(length(y))
    else
        y_LB = value.(y) .* scm.R
        σ_LB = objective_value(model) * scm.R
    end
    return (σ_LB, y_LB)
end

function solve_LB(scm::NNSCM, p, pbar, Mα=20; noise=0)
    model = scm.lp_model
    y = all_variables(model)
    # Remove all constraints
    for (F, S) in list_of_constraint_types(model)  
        for con in all_constraints(model, F, S)
            delete(model, con)
        end
    end
    # Bounding Constraints
    for i in eachindex(y)
        @constraint(model, -1 * scm.σ_maxs[i] / scm.UBs[pbar][1] / scm.R <= 
                            y[i] <= 
                            scm.σ_maxs[i] / scm.UBs[pbar][1] / scm.R)
    end
    # Stability Constraints
    C_NN_idxs, _ = knn(scm.kdtrees[pbar], p, min(Mα, length(scm.kdtrees[pbar].data)))
    for i in C_NN_idxs
        p_c = scm.kdtrees[pbar].data[i]
        ub = scm.β_UBs[pbar][p_c][1]
        if !isnan(ub)
            # In case of negative ub
            @constraint(model, J(scm,p_c,y) >= ub / scm.R)
        end
    end
    @objective(model, Min, J(scm,p,y))
    optimize!(model)
    if !(termination_status(model) in (OPTIMAL, LOCALLY_SOLVED))
        if noise >= 1
            println("Warning: Linear Program Solution not optimal; termination status: $(termination_status(model))")
            println("Setting β_LB=0")
        end
        β_LB = 0.0
        y_LB = zeros(length(y))
    else
        y_LB = value.(y) .* scm.R
        β_LB = objective_value(model) * scm.R
    end
    return (β_LB, y_LB)
end

function solve_UB(scm::AbstractSCM, p)
    σ_UB = Inf
    for (_,y) in values(scm.UBs)
        Jval = J(scm, p, y, Ap=(isa(scm,NNSCM) ? scm.AAp : scm.Ap))
        if Jval < σ_UB
            σ_UB = Jval
        end
    end
    return σ_UB
end

function solve_UB(scm::NNSCM, p, pbar)
    β_UB = Inf
    for (β,y) in values(scm.β_UBs[pbar])
        if isnan(β) || β <= 0
            continue
        end
        Jval = J(scm, p, y)
        if Jval < β_UB
            β_UB = Jval
        end
    end
    return β_UB
end

function get_all_constraints(scm::ANLSCM)
    names = String[]
    vals = Float64[]
    QAA = length(scm.Ap.arrays)
    QA = Int((-1 + sqrt(1 + 8*QAA)) / 2)
    y = all_variables(scm.nlp_model)
    idx = 1
    for i in 1:QA
        idx += 1
        for j in (i+1):QA
            name = "$i-$j"
            push!(names, name)
            idxii = idx - (j - i)
            idxjj = idx + (QA - j + 1)
            for jct in i+1:(j-1)
                idxjj += (QA - jct + 1)
            end
            con = constraint_by_name(scm.nlp_model, name)
            nl_coef = -1 * normalized_coefficient(con, y[idxii], y[idxjj])
            push!(vals, nl_coef)
            idx += 1
        end
    end
    return names, vals
end

function set_constraint(scm::ANLSCM, i::Int, j::Int, val::Float64)
    names = String[]
    vals = Float64[]
    QAA = length(scm.Ap.arrays)
    QA = Int((-1 + sqrt(1 + 8*QAA)) / 2)
    y = all_variables(scm.nlp_model)
    name = "$i-$j"
    con = constraint_by_name(scm.nlp_model, name)
    ii = 1
    for ict in 1:(i-1)
        ii += QA - ict
    end
    set_normalized_coefficient(con, y[min_ii], y[min_jj], -1 * abs(val))
end

function add_param!(scm::SCM, p; noise=0, reigkwargs...)
    σ, y = make_σ_UB(scm, p, noise=noise; reigkwargs...)
    push!(scm.UBs, p => (σ, y))
    scm.σ_LBs[p] = σ
    data = scm.kdtree.data
    push!(data, p)
    scm.kdtree = KDTree(data, reorder=false)
    nothing
end

"""
`constrain!(scm::SCM, ps::AbstractVector, ϵ::Real, Mα::Int, Mp::Int[; make_monotonic=true, max_iter=500, noise=0, reigkwargs...])`

Constrains `scm` about parameters `ps` to relative gap `ϵ`. `Mα` and `Mp` are the stability and positivity constraints
respectively. `make_monotonic` ensures LB predictions increase monotonically. `max_iter` is the maximum number of SCM 
iterations ran. `noise` determines amount of printed output. See documentation of `ModelOrderReductionToolkit.reig` for
arguments to `reigkwargs`.
"""
function constrain!(scm::SCM, ps::AbstractVector, ϵ::Real, Mα::Int, Mp::Int; make_monotonic=true, max_iter=500, noise=0, reigkwargs...)
    if noise >= 1
        println("Beginning SCM Procedure")
        println("----------")
    end
    if !(eltype(ps) <: SVector)
        P = length(ps[1])
        ps = SVector{P}.(ps)
    end
    pos_kdtree = (Mp == 0) ? KDTree(zeros(length(ps[1]), 0)) : KDTree(ps, reorder=false)
    scm.pos_kdtree = pos_kdtree
    if !make_monotonic
        empty!(scm.σ_LBs)
    end
    for _ in 1:max_iter
        ϵ_k = -1.0
        σ_UB_k = 0
        σ_LB_k = 0
        p_k = nothing
        i_k = 0
        # Loop through every point in discretization to find 
        # arg max {p in discretization} (σ_UB(p) - σ_LB(p)) / σ_UB(p)
        for (pidx,p) in enumerate(ps)
            if p in keys(scm.UBs)
                continue
            end
            ## Solve linear program to find σ_LB, y_LB
            σ_LB, y_LB = solve_LB(scm, p, Mα, Mp, noise=noise)
            if σ_LB > get(scm.σ_LBs, p, -Inf)
                scm.σ_LBs[p] = σ_LB
            end
            ## Loop through Y_UB to find σ_UB
            σ_UB = solve_UB(scm, p)
            ## Compute ϵ_k
            ϵ_disc = 1 - σ_LB / σ_UB
            if !scm.coercive
                ϵ_disc = (ϵ_disc <= 1) ? 1-sqrt(1-ϵ_disc) : ϵ_disc
            end
            if ϵ_disc > ϵ_k
                if noise >= 2
                    @printf("Updating ϵ_k to %.4e\n",ϵ_disc)
                    @printf("p = %s\n",p)
                    @printf("UB is %.6f\n",σ_UB)
                    @printf("LBs are %.6f, %s\n",σ_LB,y_LB)
                end 
                ϵ_k = ϵ_disc
                p_k = p
                i_k = pidx
                σ_UB_k = σ_UB
                σ_LB_k = σ_LB
            end
        end
        if isnothing(p_k)
            if noise >= 1
                println("Warning: Looped through all of parameter discretization without meeting ϵ bound")
            end
            return
        end
        if noise >= 1
            @printf("k = %d, ϵ_k = %.4e, σ_UB_k = %.4f, σ_LB_k = %.4f, p_k = %s\n", 
                    length(keys(scm.UBs)), ϵ_k, σ_UB_k, σ_LB_k, p_k)
        end
        if ϵ_k < ϵ# && isnothing(adaptive_nl_rate)
            if noise >= 1
                println("----------")
                @printf("Terminating on iteration k = %d with ϵ_k=%.4e\n",
                        length(keys(scm.UBs)),ϵ_k)
            end
            return
        end
        # Update for next loop
        add_param!(scm, p_k; reigkwargs...)
    end
end

function adapt_constraints!(scm::ANLSCM, p, σ_LB, y_LB, σ_UB, Mα; adaptive_nl_rate=2.0, adaptive_eps_tol=1e-6, noise=0)
    QAA = length(scm.Ap.arrays)
    QA = Int((-1 + sqrt(1 + 8*QAA)) / 2)
    y = all_variables(scm.nlp_model)
    ϵ_here = (1 - σ_LB / σ_UB)
    adaptive_eps_tol = -1 * abs(adaptive_eps_tol)
    did_adapt = false
    obj_fun = objective_function(scm.nlp_model)
    while ϵ_here < adaptive_eps_tol
        if noise >= 2
            println("Found negative ϵ=$ϵ_here at p=$p, adapting NL constraints")
        end
        minactivity = Inf
        mincon = nothing
        min_nl_coef = 0.0
        min_i = 1
        min_ii = 1
        min_j = 2
        min_jj = 2
        idx = 1
        for i in 1:QA
            idx += 1
            for j in (i+1):QA
                idxii = idx - (j - i)
                idxjj = idx + (QA - j + 1)
                for jct in i+1:(j-1)
                    idxjj += (QA - jct + 1)
                end
                con = constraint_by_name(scm.nlp_model, "$i-$j")
                nl_coef = -1 * normalized_coefficient(con, y[idxii], y[idxjj])
                activity = nl_coef * abs(y_LB[idxii] * y_LB[idxjj]) - y_LB[idx]^2#nl_coef - y_LB[idx]^2 / y_LB[idxii]*y_LB[idxjj]
                if iszero(coefficient(obj_fun, y[idxii])) || iszero(coefficient(obj_fun, y[idxjj]))
                    activity = Inf
                end
                #activity = nl_coef * abs(y_LB[idxii] * y_LB[idxjj]) / max(1e-2,y_LB[idx]^2) - 1
                if nl_coef < 1.0 && activity < minactivity
                    minactivity = activity
                    mincon = con
                    min_nl_coef = nl_coef
                    min_i = i; min_j = j
                    min_ii = idxii; min_jj = idxjj
                end
                idx += 1
            end
        end
        if isnothing(mincon)
            # No constraints to loosen
            break
        end
        if min_nl_coef >= 1.0
            if noise >= 1
                println("exiting b/c old coef=$min_nl_coef")
            end
            # No adapting to do
            break
        end
        # Adapt closest to active constraint
        new_coef = 1 - (1 - min_nl_coef) / adaptive_nl_rate
        # println("adaptive rate is $adaptive_nl_rate")
        # println("old is $min_nl_coef")
        # println("new_coef is $new_coef")
        if new_coef >= 1.0 || new_coef ≈ 1.0
            if noise >= 1
                println("exiting b/c new_coef=$new_coef")
            end
            set_normalized_coefficient(mincon, y[min_ii], y[min_jj], -1.0)
            break
        end
        set_normalized_coefficient(mincon, y[min_ii], y[min_jj], -1 * new_coef)
        did_adapt = true
        if noise == 1
            println("Adapting $min_i-$min_j nonlinear constraint")
        elseif noise >= 2
            println("Adapting $min_i-$min_j nonlinear constraint to y[ij]^2 ≤ $new_coef * y[ii] * y[jj]")
        end
        ## Solve linear program to find σ_LB, y_LB
        σ_LB, y_LB = solve_LB(scm, p, Mα, noise=noise)
        ϵ_here = (1 - σ_LB / σ_UB)
        if noise >= 1
            println("New ϵ=$ϵ_here, p=$p (break condition $(!(ϵ_here < adaptive_eps_tol)))")
        end
    end
    return σ_LB, y_LB, did_adapt
end

function add_param!(scm::ANLSCM, p; noise=0, reigkwargs...)
    σ, y = make_σ_UB(scm, p, noise=noise; reigkwargs...)
    push!(scm.UBs, p => (σ, y))
    data = collect(scm.kdtree.data)
    push!(data, p)
    scm.kdtree = KDTree(data, reorder=false)
    nothing
end

"""
`constrain!(scm::ANLSCM, ps::AbstractVector, ϵ::Real, Mα::Int[; adaptive_nl_rate=1.1, adaptive_eps_tol=1e-6, max_iter=500, noise=0, reigkwargs...])`

Constrains `scm` about parameters `ps` to relative gap `ϵ`. `Mα` is the stability constraint. 
`adaptive_nl_rate` and `adaptive_eps_tol` determine when to and rate at which to loosen nonlinear constraints. 
Set `adaptive_nl_rate=nothing` for no adaptive updating.
`max_iter` is the maximum number of SCM iterations ran. `noise` determines amount of printed output.
See `ModelOrderReductionToolkit.reig` for arguments to `reigkwargs`.
"""
function constrain!(scm::ANLSCM, ps::AbstractVector, ϵ::Real, Mα::Int; adaptive_nl_rate=1.1, adaptive_eps_tol=1e-6, max_iter=500, noise=0, reigkwargs...)
    if noise >= 1
        println("Beginning SCM Procedure")
        println("----------")
    end
    if !(eltype(ps) <: SVector)
        P = length(ps[1])
        ps = SVector{P}.(ps)
    end
    for _ in 1:max_iter
        ϵ_k = -1.0
        σ_UB_k = 0
        σ_LB_k = 0
        p_k = nothing
        i_k = 0
        adapted = false
        # Loop through every point in discretization to find 
        # arg max {p in discretization} (σ_UB(p) - σ_LB(p)) / σ_UB(p)
        for (pidx,p) in enumerate(ps)
            ## Solve linear program to find σ_LB, y_LB
            σ_LB, y_LB = solve_LB(scm, p, Mα, noise=noise)
            ## Loop through Y_UB to find σ_UB
            σ_UB = solve_UB(scm, p)
            ## Adapt if necessary
            if !isnothing(adaptive_nl_rate)
                σ_LB, y_LB, did_adapt = adapt_constraints!(scm, p, σ_LB, y_LB, σ_UB, Mα, adaptive_nl_rate=adaptive_nl_rate, adaptive_eps_tol=adaptive_eps_tol, noise=noise)
                adapted |= did_adapt
            end
            if σ_LB > get(scm.σ_LBs, p, -Inf)
                scm.σ_LBs[p] = σ_LB
            end
            if p in keys(scm.UBs)
                continue
            end
            ## Compute ϵ_k
            ϵ_disc = 1 - σ_LB / σ_UB
            ϵ_disc = (ϵ_disc <= 1) ? 1-sqrt(1-ϵ_disc) : ϵ_disc
            if ϵ_disc > ϵ_k
                if noise >= 2
                    @printf("Updating ϵ_k to %.4e\n",ϵ_disc)
                    @printf("p = %s\n",p)
                    @printf("UB is %.6f\n",σ_UB)
                    @printf("LBs are %.6f, %s\n",σ_LB,y_LB)
                end 
                ϵ_k = ϵ_disc
                p_k = p
                i_k = pidx
                σ_UB_k = σ_UB
                σ_LB_k = σ_LB
            end
        end
        if isnothing(p_k)
            if noise >= 1
                println("Warning: Looped through all of parameter discretization without meeting ϵ bound")
            end
            return
        end
        if noise >= 1
            @printf("k = %d, ϵ_k = %.4e, σ_UB_k = %.4f, σ_LB_k = %.4f, p_k = %s\n", 
                    length(keys(scm.UBs)), ϵ_k, σ_UB_k, σ_LB_k, p_k)
        end
        if ϵ_k < ϵ && !adapted
            if noise >= 1
                println("----------")
                @printf("Terminating on iteration k = %d with ϵ_k=%.4e and no adaptions necessary\n",
                        length(keys(scm.UBs)),ϵ_k)
            end
            return
        elseif ϵ_k < ϵ && noise >= 1
            @printf("On iteration k = %d with ϵ_k=%.4e and but adaptions were necessary\n",
                    length(keys(scm.UBs)),ϵ_k)
        end
        # Update for next loop
        add_param!(scm, p_k; reigkwargs...)
        σ_k, y_k = scm.UBs[p_k]
        # Check if need to adapt
        if !isnothing(adaptive_nl_rate) && σ_LB_k > σ_k + adaptive_eps_tol
            σ_LB, y_LB, did_adapt = adapt_constraints!(scm, p_k, σ_k, y_LB, σ_k, Mα, adaptive_nl_rate=adaptive_nl_rate, adaptive_eps_tol=adaptive_eps_tol, noise=noise)
            adapted |= did_adapt
        end
    end
end

function next_pbar(scm::NNSCM, ps::AbstractVector, ps_left=trues(length(ps)), ϵ=0.8, Mα=20; pruning=false, noise=0)
    max_ϵ = -Inf
    pbar = nothing;
    i_opt = nothing;
    for (i,p) in enumerate(ps)
        if !ps_left[i] || p in keys(scm.β_UBs)
            ps_left[i] = false
            continue
        end
        σ_UB_p = sqrt(solve_UB(scm, p))
        σ_LB_p = get(scm.σ_LBs, p, -Inf)
        if isinf(σ_LB_p)
            for pbar in keys(scm.UBs)
                σ_pbar = scm.UBs[pbar][1]
                β_LB_pbar, _ = solve_LB(scm, p, pbar, Mα, noise=noise)
                σ_LB_pbar = β_LB_pbar*σ_pbar
                if σ_LB_pbar > σ_LB_p
                    σ_LB_p = σ_LB_pbar
                end
            end
        end
        if σ_LB_p > get(scm.σ_LBs, p, -Inf)
            scm.σ_LBs[p] = σ_LB_p
        end
        ϵ_p = σ_UB_p == Inf ? Inf : 1 - σ_LB_p / σ_UB_p
        if ϵ_p > max_ϵ
            i_opt = i
            pbar = p
            max_ϵ = ϵ_p
        end
        if pruning && ϵ_p <= ϵ
            # p meets desired ϵ bound
            ps_left[i] = false
            continue
        end
    end
    if !isnothing(pbar)
        ps_left[i_opt] = false
    end
    return (pbar, max_ϵ)
end

function add_pbar!(scm::NNSCM, pbar; noise=0, reigkwargs...)
    P = length(pbar)
    pbar = isa(pbar, SVector) ? pbar : SVector{P}(pbar)
    σ², y = make_σ_UB(scm, pbar)
    scm.UBs[pbar] = (sqrt(σ²), y)
    β, y = make_β_UB(scm, pbar, pbar, noise=noise; reigkwargs...)
    scm.β_UBs[pbar] = Dict(pbar => (β,y))
    scm.β_LBs[pbar] = Dict(pbar => β)
    scm.σ_LBs[pbar] = sqrt(σ²)
    scm.kdtrees[pbar] = KDTree([pbar])
    nothing
end

function add_param!(scm::NNSCM, p, pbar; noise=0, reigkwargs...)
    β, y = make_β_UB(scm, p, pbar, noise=noise; reigkwargs...)
    σ_pbar = scm.UBs[pbar][1]
    if β * σ_pbar > get(scm.σ_LBs, p, -Inf)
        scm.σ_LBs[p] = β * σ_pbar
    end
    scm.β_UBs[pbar][p] = (β, y)
    scm.β_LBs[pbar][p] = β
    if β > 1e-10
        data = scm.kdtrees[pbar].data
        push!(data, p)
        scm.kdtrees[pbar] = KDTree(data, reorder=false)
    end
    nothing
end

function remove_pbars!(scm::NNSCM, ps::AbstractVector, ϵ_keep=0.2; noise=0)
    pbars = collect(keys(scm.β_UBs))
    opt_eps = falses(length(pbars), length(ps))
    for (i,p) in enumerate(ps)
        σ_UB = scm(p, which=:U)
        for (j,pbar) in enumerate(pbars)
            σ_LB = scm(p, pbar=pbar)
            if 1 - σ_LB / σ_UB <= ϵ_keep
                opt_eps[j,i] = true
            end
        end
    end
    keep = trues(length(pbars))
    for j in reverse(eachindex(pbars))
        pbar = pbars[j]
        idxs = [i for i in eachindex(pbars) if i!=j && keep[i]]
        if all(sum(opt_eps[idxs,:], dims=1) .> 0)
            keep[j] = false
            delete!(scm.β_UBs, pbar)
            delete!(scm.UBs, pbar)
            delete!(scm.kdtrees, pbar)
            delete!(scm.β_LBs, pbar)
            if noise >= 1
                @printf("Removed pbar=%s\n", pbar)
            end
        end
    end
    return 
end

"""
`constrain!(scm::NNSCM, ps::AbstractVector, pbar::AbstractVector[, ϵ_β=0.8, ϕ=0.0, Mα=20, ps_left=trues(length(ps)); max_iter=500, p_choice=3, noise=0, reigkwargs...])`

Constrains the `pbar`-domain decomposition of the `scm` about parameters `ps`. `ϵ_β` determines the natural-norm relative gap, `ϕ`
determines the minimum LB prediction required for inclusion in domain, `Mα` is the stability parameter, `ps_left` is a vector of
which parameters to consider, `max_iter` is maximum number of iterations.

`p_choice` can be 1, 2, or 3. 
- `p_choice=1` - Only add parameter with highest natural norm relative gap
- `p_choice=2` - Only add parameter within domain (`LB ≥ ϕ`) with highest natural norm relative gap
- `p_choice=3` - Do both of above

See `ModelOrderReductionToolkit.reig` for arguments to `reigkwargs`.
"""
function constrain!(scm::NNSCM, ps::AbstractVector, pbar::AbstractVector, ϵ_β=0.8, ϕ=0.0, Mα=20, ps_left=trues(length(ps)); max_iter=500, p_choice=3, noise=0, reigkwargs...)
    current_domain = Set([pbar])
    β_UBs = scm.β_UBs[pbar]
    β_LBs = scm.β_LBs[pbar]
    σ_pbar = scm.UBs[pbar][1]
    for _ in 1:max_iter
        prev_domain = copy(current_domain)
        empty!(current_domain) # In case points have left domain
        ϵ_β_k = -Inf
        ϵ_β_domain = -Inf
        β_UB_k = 0
        β_UB_domain = 0
        β_LB_k = 0
        β_LB_domain = 0
        p_k = nothing
        p_domain = nothing
        # Loop through every point in discretization to find 
        # arg max {p in discretization} (β_UB(p) - β_LB(p)) / β_UB(p)
        for (i,p) in enumerate(ps)
            if get(β_UBs, p, (-Inf,nothing))[1] > 0
                push!(current_domain, p)
                continue
            elseif p in keys(β_UBs)
                continue
            end
            if !ps_left[i]
                continue
            end
            ## Solve linear program to find β_LB, y_LB
            β_LB, y_LB = solve_LB(scm, p, pbar, Mα, noise=noise)
            σ_LB = β_LB * σ_pbar
            if σ_LB > get(scm.σ_LBs, p, -Inf)
                scm.σ_LBs[p] = σ_LB
            end
            if β_LB > ϕ
                push!(current_domain, p)
            end
            if β_LB >= get(β_LBs, p, 0.0)
                β_LBs[p] = β_LB
            end
            ## Loop through Y_UB to find β_UB
            β_UB = solve_UB(scm, p, pbar)
            if β_UB <= 1e-10
                # Avoid issue at β_UB ≈ 0.0
                continue
            end
            ## Compute ϵ_k
            ϵ_β_disc = 1 - β_LB / β_UB
            if ϵ_β_disc > ϵ_β_k
                if noise >= 2
                    @printf("Updating ϵ_β_k to %.4e\n", ϵ_β_disc)
                    @printf("p = %s\n",p)
                    @printf("UB is %.6f\n",β_UB)
                    @printf("LBs are %.6f, %s\n",β_LB,y_LB)
                end 
                ϵ_β_k = ϵ_β_disc
                p_k = p
                β_UB_k = β_UB
                β_LB_k = β_LB
            end
            if p in current_domain && ϵ_β_disc > ϵ_β_domain
                ϵ_β_domain = ϵ_β_disc
                p_domain = p
                β_UB_domain = β_UB
                β_LB_domain = β_LB
            end
        end
        new_domain_size = length(current_domain)
        if isnothing(p_k)
            if noise >= 1
                println("Warning: Selected all parameters in domain (size $(new_domain_size)), beginning new domain")
            end
            # Go to next parameter decomposition
            break
        end
        if noise >= 1
            @printf("k = %d, ϵ_β_k = %.4e, ϵ_β_domain = %.4e, domain_size = %d, p_k = %s\n", 
                    length(keys(β_UBs)), ϵ_β_k, ϵ_β_domain, new_domain_size, p_k)
        end
        if length(keys(β_UBs)) > 1 && ϵ_β_domain < ϵ_β && new_domain_size <= length(prev_domain)
            if noise >= 1
                @printf("Exiting domain with k = %d, domain size %d\n",
                        length(keys(β_UBs)), new_domain_size)
            end
            # Go to next parameter decomposition
            break
        end
        # Update for next loop
        if p_choice in (1,3)
            # Add param with highest ϵ_β
            add_param!(scm, p_k, pbar, noise=noise; reigkwargs...)
        end
        if !isnothing(p_domain) && p_domain != p_k && ϵ_β_domain > ϵ_β && p_choice in (2,3)
            # Add param in domain with highest ϵ_β
            add_param!(scm, p_domain, pbar, noise=noise; reigkwargs...)
        end
    end
end

"""
`constrain!(scm::NNSCM, ps::AbstractVector, ϵ::Real[, ϵ_β=0.8, ϕ=0.0, Mα=20; pruning=false, removals=false, eps_keep=0.2, p_choice=3, max_iter=500, max_inner_iter=500, noise=0, reigkwargs...])`

Constrains `scm` about parameters `ps` to relative gap `ϵ` with inner-maximum relative gap `ϵ_β` and domain-inclusion 
parameter `ϕ`. `Mα` is the stability constraint. `pruning` determines whether or not to exclude parameters from consideration
when their relative gap decreases below `ϵ` for a greedier procedure. `removals` determines whether or not to try removing
unnecessary domains at the start of each iteration with tolerance `eps_keep`. See other `constrain!` method for `p_choice`.
`max_iter` is the maximum number of SCM iterations ran and `max_inner_iter` determines the maximum number of iterations
per domain decomposition. `noise` determines amount of printed output.
"""
function constrain!(scm::NNSCM, ps::AbstractVector, ϵ::Real, ϵ_β=0.8, ϕ=0.0, Mα=20; pruning=false, removals=false, eps_keep=0.2, p_choice=3, max_iter=500, max_inner_iter=500, noise=0, reigkwargs...)
    if noise >= 1
        println("Beginning NNSCM Procedure")
    end
    if !(eltype(ps) <: SVector)
        P = length(ps[1])
        ps = SVector{P}.(ps)
    end
    ps_left = trues(length(ps))
    for _ in 1:max_iter
        if noise >= 1
            println("----------")
        end
        if removals
            remove_pbars!(scm, ps, eps_keep, noise=noise)
        end
        # Select new parameter pbar to decompose domain about
        pbar, ϵ_pbar = next_pbar(scm, ps, ps_left, ϵ, Mα, pruning=pruning, noise=noise)
        if isnothing(pbar) 
            if noise >= 1
                println("All parameters selected, exiting")
            end
            return
        end
        if ϵ_pbar < ϵ
            if noise >= 1
                @printf("pbar ϵ value = %.2e < %.2e, returning\n", ϵ_pbar, ϵ)
            end
            return
        end
        add_pbar!(scm, pbar, noise=noise; reigkwargs...)
        if noise >= 1
            @printf("Domain decomposition number %d: ϵ = %.4e, p_bar = %s\n", 
                        length(keys(scm.β_UBs)), ϵ_pbar, pbar)
        end
        # Perform NNSCM about pbar
        constrain!(scm, ps, pbar, ϵ_β, ϕ, Mα, ps_left, max_iter=max_inner_iter,
                   p_choice=p_choice, noise=noise; reigkwargs...)
    end
end

function (scm::SCM)(p, Mα=20, Mp=0; which=:L, noise=0)
    if which == :U
        σ_UB = solve_UB(scm, p)
        if !scm.coercive
            σ_UB = sqrt(max(0.0,σ_UB))
        end
        return σ_UB
    elseif which == :E
        return 1 - scm(p, Mα, Mp, which=:L, noise=noise) / scm(p, Mα, Mp, which=:U, noise=noise)
    elseif which == :L
        if isa(p, Number)
            p = SVector(p)
        end
        σ_LB = get(scm.σ_LBs, p, begin
            σ_LB_p = solve_LB(scm, p, Mα, Mp, noise=noise)[1] 
            scm.σ_LBs[p] = σ_LB_p
            σ_LB_p
        end)
        if !scm.coercive
            σ_LB = sqrt(max(0.0,σ_LB))
        end
        return σ_LB
    else
        error("Unknown which=$which, choose from (:L, :U, :E)")
    end
end

function (scm::ANLSCM)(p, Mα=20; which=:L, sq=false, noise=0)
    if which == :U
        σ_UB = solve_UB(scm, p)
        if !sq
            σ_UB = sqrt(max(0.0,σ_UB))
        end
        return σ_UB
    elseif which == :E
        return 1 - scm(p, Mα, which=:L, sq=sq, noise=noise) / scm(p, Mα, which=:U, sq=sq, noise=noise)
    elseif which == :L
        if isa(p, Number)
            p = SVector(p)
        end
        σ_LB = get(scm.σ_LBs, p, begin
            σ_LB_p = solve_LB(scm, p, Mα, noise=noise)[1] 
            scm.σ_LBs[p] = σ_LB_p
            σ_LB_p
        end)
        if !sq
            σ_LB = sqrt(max(0.0,σ_LB))
        end
        return σ_LB
    else
        error("Unknown which=$which, choose from (:L, :U, :E)")
    end
end

function (scm::NNSCM)(p, Mα=20; pbar=nothing, which=:L, noise=0)
    if which == :U
        return sqrt(solve_UB(scm, p))
    elseif which == :E
        return 1 - scm(p, Mα, pbar=pbar, which=:L, noise=noise) / scm(p, Mα, pbar=pbar, which=:U, noise=noise)
    elseif which == :W
        σ_LB = 0.0
        thepbar = nothing
        for pbar in keys(scm.UBs)
            σ_pbar = scm.UBs[pbar][1]
            β_LB_pbar = solve_LB(scm, p, pbar, Mα, noise=noise)[1]
            σ_LB_pbar = σ_pbar * β_LB_pbar
            if σ_LB_pbar > σ_LB
                σ_LB = σ_LB_pbar
                thepbar = pbar
            end
        end
        return thepbar
    elseif which == :L
        σ_LB = 0.0
        if isa(p, Number)
            p = SVector(p)
        end
        if !isnothing(pbar) # pbar provided
            σ_pbar = scm.UBs[pbar][1]
            β_LB_pbar = solve_LB(scm, p, pbar, Mα, noise=noise)[1]
            σ_LB = σ_pbar * β_LB_pbar
        elseif p in keys(scm.σ_LBs)
            σ_LB = scm.σ_LBs[p]
        else
            for pbar in keys(scm.UBs)
                σ_pbar = scm.UBs[pbar][1]
                β_LB_pbar = solve_LB(scm, p, pbar, Mα, noise=noise)[1]
                σ_LB_pbar = σ_pbar * β_LB_pbar
                if σ_LB_pbar > σ_LB
                    σ_LB = σ_LB_pbar
                end
            end
            scm.σ_LBs[p] = σ_LB
        end
        return σ_LB
    else
        error("Unknown which=$which, choose from (:L, :U, :E)")
    end
end