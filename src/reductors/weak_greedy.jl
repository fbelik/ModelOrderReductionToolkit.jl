"""
`ErrorEstimator`

Abstract type for error estimation of a model
used for weak greedy MOR.
"""
abstract type ErrorEstimator end

function approximate_error(estimator::ErrorEstimator, x_r::AbstractVector, p, i::Int=1)
    error("Must implement approximate_error(estimator, x_r, p, i::Int=1) for ErrorEstimator")
end

function update_estimator!(estimator::ErrorEstimator, x_r::AbstractVector)
    error("Must implement update_estimator!(estimator, x_r) for ErrorEstimator")
end

"""
`StabilityResidualErrorEstimator <: ErrorEstimator`

Error estimator for a `LinearModel` which approximates the 
stability factor with `stability_estimator(p)` and computes 
the residual norm with `residual_norm`.
"""
struct StabilityResidualErrorEstimator <: ErrorEstimator
    stability_estimator::Function
    residual_norms::Vector{ResidualNormComputer}
    function StabilityResidualErrorEstimator(model::LinearModel, custom_stability_estimator::Function; residual=:proj)
        residual_norm = begin
            if residual == :standard
                StandardResidualNormComputer(model.Ap, model.bp)
            elseif residual == :proj
                ProjectionResidualNormComputer(model.Ap, model.bp)
            else
                error("Unknown residual keyword: $residual; choose from (:standard, :proj)")
            end
        end
        return new(custom_stability_estimator, [residual_norm])
    end
    function StabilityResidualErrorEstimator(model::LinearMatrixModel, custom_stability_estimator::Function; residual=:proj)
        residual_norms = [begin
            if residual == :standard
                StandardResidualNormComputer(model.Ap, bp)
            elseif residual == :proj
                ProjectionResidualNormComputer(model.Ap, bp)
            else
                error("Unknown residual keyword: $residual; choose from (:standard, :proj)")
            end
        end for bp in model.bps]
        return new(custom_stability_estimator, residual_norms)
    end
end

function Base.show(io::Core.IO, estimator::StabilityResidualErrorEstimator)
    println(io, "Stability Residual Error Estimator")
    print(io, "Stability method - ")
    println(io, estimator.stability_estimator)
end

"""
`approximate_error(estimator, x_r, p)`

For a `StabilityResidualErrorEstimator`, computes the residual
norm `||b(p) - A(p) V x_r(p)||` and the stability factor `σ_min(A(p))`,
to approximate the error
`||x(p) - V x_r(p)|| <≈ ||b(p) - A(p) V x_r(p)|| / σ_min(A(p))`.
"""
function approximate_error(estimator::StabilityResidualErrorEstimator, x_r::AbstractVector, p, i::Int=1)
    residual_norm = compute(estimator.residual_norms[i], x_r, p)
    stability_factor = estimator.stability_estimator(p)
    return residual_norm / stability_factor
end

"""
`update_estimator!(estimator, x)`

For a `StabilityResidualErrorEstimator`, once a full, orthonormalized,
vector `x` is formed, updates the `residual_norm` object. 
"""
function update_estimator!(estimator::StabilityResidualErrorEstimator, x::AbstractVector)
    for residual_norm in estimator.residual_norms
        update!(residual_norm, x)
    end
end

"""
`wg_reductor <: WGReductor`

Stores a FOM in `model`, an error estimator `estimator`, 
the greedily selected parameters `params_greedy`, the reduced
basis in `V`, the ROM in `rom`, the approximate errors at each
step in `approx_errors`, and the truth errors in each step at
`truth_errors`.
"""
struct WGReductor{NOUT}
    model::StationaryModel{NOUT}
    estimator::ErrorEstimator
    params_greedy::AbstractVector{Set}
    V::VectorOfVectors
    rom::StationaryModel
    approx_errors::Vector{Float64}
    truth_errors::Vector{Float64}
end

function Base.show(io::Core.IO, reductor::WGReductor)
    res  = "WG reductor with RB dimension $(size(reductor.V,2))"
    println(io, res)
    print(io, "FOM: ")
    println(io, reductor.model)
    print(io, "ROM: ")
    println(io, reductor.rom)
    print(io, "Increase RB dimension with add_to_rb!(reductor, params) or add_to_rb!(reductor, params, r)")
end

"""
`wg_reductor = WGReductor(model, estimator[; noise=1])`

Given an `ErrorEstimator`, `estimator`, and a `StationaryModel`, `model`, 
initializes a `WGReductor` object with a null reduced basis.
"""
function WGReductor(model::StationaryModel{NOUT}, estimator::ErrorEstimator; noise=1) where NOUT
    V = VectorOfVectors(output_length(model), 0, output_type(model))
    rom = galerkin_project(model, V)
    params_greedy = [Set() for _ in 1:NOUT]
    return WGReductor{NOUT}(model, estimator, params_greedy, V, rom, Float64[], Float64[])
end

"""
`add_directly_to_rb(wg_reductor, x)`

Assumes that `x` is already orthonormalized with respect to
`wg_reductor.V`, and adds `x` to the RB. Additionally, updates
the error estimator. Typically not to be called externally.
"""
function add_directly_to_rb!(wg_reductor::WGReductor, x::AbstractVector)
    galerkin_add!(wg_reductor.rom, wg_reductor.model, x, wg_reductor.V)
    addCol!(wg_reductor.V)
    wg_reductor.V[:, end] .= x
    update_estimator!(wg_reductor.estimator, x)
end

"""
`add_to_rb!(wg_reductor, params[; noise=0, progress=false, eps=0.0, zero_tol=1e-15])`

Loops through the vector of parameters `params`, computes the approximate estimator
for each, selects the one with the highest error, and updates `wg_reductor` with 
the corresponding full order solution. Returns `true` if a vector is added to the RB,
`false` otherwise.
"""
function add_to_rb!(wg_reductor::WGReductor{NOUT}, params::AbstractVector; noise=0, progress=false, eps=0.0, zero_tol=1e-15) where NOUT
    k = size(wg_reductor.V, 2) + 1
    if k > output_length(wg_reductor.model)
        if noise >= 1
            println("Cannot add more than $(k-1) snapshots, not appending to RB")
        end
        return false
    end
    max_error = -1.0
    max_i = -1
    max_p = nothing
    for p in (progress ? ProgressBar(params) : params)
        for i in 1:NOUT
            if p in wg_reductor.params_greedy[i]
                continue
            end
            x_r = wg_reductor.rom(p, i)
            approx_error = approximate_error(wg_reductor.estimator, x_r, p, i)
            if approx_error > max_error
                max_error = approx_error
                max_i = i
                max_p = p
            end
        end
    end
    if max_p == nothing
        if noise >= 1
            println("($k) No new parameters with non-null error encountered, not adding snapshot")
        end
        return false
    end
    push!(wg_reductor.params_greedy[max_i], max_p)
    if max_error < eps
        if noise >= 1
            @printf("(%d) approximate error = %.4e < ϵ, not adding snapshot to RB\n",k,max_error)
        end
        return false
    end
    push!(wg_reductor.approx_errors, max_error)
    # Compute full solution to add to RB
    x = wg_reductor.model(max_p, max_i)
    x_approx = lift(wg_reductor, wg_reductor.rom(max_p, max_i))
    truth_error = norm(x .- x_approx)
    push!(wg_reductor.truth_errors, truth_error)
    # Orthonormalize solution w.r.t. V
    nx = orthonormalize_mgs2!(x, wg_reductor.V)
    if nx >= zero_tol
        add_directly_to_rb!(wg_reductor, x)
    elseif noise >= 1
        @printf("After orthogonalizing, truth vector had norm %.2e < zero_tol, not appending to RB\n", nx)
        return false
    end
    if noise >= 1
        @printf("(%d) truth error = %.4e, upperbound error = %.4e\n",k,truth_error,max_error)
    end
    return true
end

"""
`add_to_rb!(wg_reductor, params, r[; noise=0, eps=0.0, zero_tol=1e-15])`

Adds to `wg_reductor` at least `r` times by calling 
`add_to_rb!(wg_reductor, params, noise=noise, eps=eps, zero_tol=zero_tol)` several times.
If all `r` are added, returns `true`, otherwise `false`.
"""
function add_to_rb!(wg_reductor::WGReductor, params::AbstractVector, r; noise=0, progress=false, eps=0.0, zero_tol=1e-15)
    for i in 1:r
        added = add_to_rb!(wg_reductor, params, noise=noise, progress=progress, eps=eps, zero_tol=zero_tol)
        if !added
            return false
        end
    end
    return true
end

"""
`form_rom(wg_reductor, r=-1)`

Calls `galerkin_project` on the FOM and returns
a ROM with RB of dimension `r`. If `r=-1`, uses
all available columns of `wg_reductor.V`.
"""
function form_rom(wg_reductor::WGReductor, r=-1)
    V = wg_reductor.V
    if r == -1 || r == size(V, 2)
        return get_rom(wg_reductor)
    elseif r > size(V, 2)
        add_to_rb!(wg_reductor, params)
    end
    if (size(V, 2) < r)
        error("r=$r must be ≤ $(size(V, 2)), first call add_to_rb! to increase reductor's RB dimension.")
    end
    return galerkin_project(wg_reductor.model, Matrix(V), r=r)
end

"""
`get_rom(wg_reductor)`

Helper method for getting the ROM from the 
`wg_reductor` object. Can otherwise obtain
it through `wg_reductor.rom`.
"""
function get_rom(wg_reductor::WGReductor)
    return wg_reductor.rom
end

"""
`lift(wg_reductor, x_r)`

Given a solution array `x_r` to a ROM formed by the
`wg_reductor` lifts the solution(s) to the same dimension of
the FOM. 
"""
function lift(wg_reductor::WGReductor, x_r::AbstractArray)
    r = size(x_r,1)
    V = wg_reductor.V
    return V[:, 1:r] * x_r
end