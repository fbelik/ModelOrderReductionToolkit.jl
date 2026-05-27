abstract type PrimalDualErrorEstimator <: ErrorEstimator end

struct FengBennerErrorComponents
    total::Float64
    residual_dual::Float64
    dual::Float64
    input::Int
    output::Int
end

mutable struct FengBennerErrorEstimator <: PrimalDualErrorEstimator
    model::LTIModel
    primal_model::LinearMatrixModel
    dual_model::LinearMatrixModel
    V::VectorOfVectors
    Vdu::VectorOfVectors
    Vrdu::VectorOfVectors
    primal_rom::LinearMatrixModel
    dual_rom::LinearMatrixModel
    rdu_rom::LinearMatrixModel
    logfreq::Bool

    function FengBennerErrorEstimator(
        model::LTIModel;
        V=nothing,
        Vdu=nothing,
        Vrdu=nothing,
        force_rb_real::Bool=false,
        logfreq::Bool=false,
        primal_model::Union{LinearMatrixModel,Nothing}=nothing,
        dual_model::Union{LinearMatrixModel,Nothing}=nothing,
    )
        n = size(model.A, 1)
        T = _transfer_basis_type(model, force_rb_real)
        Vv = _transfer_basis_as_vov(V, n, T, "V")
        Vduv = _transfer_basis_as_vov(Vdu, n, T, "Vdu")
        Vrduv = _transfer_basis_as_vov(Vrdu, n, T, "Vrdu")

        primal = isnothing(primal_model) ? transfer_primal_model(model; logfreq) : primal_model
        dual = isnothing(dual_model) ? transfer_dual_model(model; logfreq) : dual_model

        return new(
            model,
            primal,
            dual,
            Vv,
            Vduv,
            Vrduv,
            galerkin_project(primal, Vv),
            galerkin_project(dual, Vduv),
            galerkin_project(dual, Vrduv),
            logfreq,
        )
    end
end

function Base.show(io::IO, estimator::FengBennerErrorEstimator)
    print(io,
        "Feng-Benner transfer-function error estimator with dimensions ",
        "dim(V)=", size(estimator.V, 2), ", ",
        "dim(Vdu)=", size(estimator.Vdu, 2), ", ",
        "dim(Vrdu)=", size(estimator.Vrdu, 2),
    )
end

# -----------------------------------------------------------------------------
# Sample and type handling
# -----------------------------------------------------------------------------

function _transfer_basis_type(model::LTIModel, force_rb_real::Bool)
    T = promote_type(
        Float64,
        eltype(model.A),
        eltype(model.B),
        eltype(model.C),
        eltype(model.D),
    )
    return force_rb_real ? real(T) : complex(T)
end

function _transfer_basis_as_vov(V, n::Int, ::Type{T}, name::AbstractString) where T
    if isnothing(V)
        return VectorOfVectors(n, 0, T)
    end
    size(V, 1) == n || throw(DimensionMismatch("$name must have $n rows"))
    if V isa VectorOfVectors{T}
        return V
    end
    return VectorOfVectors(Matrix{T}(V))
end

function _transfer_parameter(sample)
    sample isa Real && return [sample]
    if sample isa Tuple && length(sample) == 2 && sample[1] isa Real
        omega = sample[1]
        phys = sample[2]
        isnothing(phys) && return [omega]
        phys isa AbstractVector && return [omega, phys...]
        return [omega, phys]
    end
    sample isa AbstractVector && return sample
    return collect(sample)
end

# -----------------------------------------------------------------------------
# Frequency-domain LinearMatrixModels
# -----------------------------------------------------------------------------

function transfer_primal_model(model::LTIModel; logfreq::Bool=false)
    return to_frequency_domain(model, logfreq)
end

function transfer_dual_model(
    model::LTIModel,
    logfreq::Bool=false,
)
    return to_frequency_domain(adjoint(model), logfreq)
end

# -----------------------------------------------------------------------------
# Reduced model updates
# -----------------------------------------------------------------------------

_as_matrix_block(X::AbstractVector) = reshape(X, :, 1)
_as_matrix_block(X::AbstractMatrix) = X

function _append_basis_block!(V::VectorOfVectors{T}, X::AbstractVecOrMat) where T
    for x in eachcol(_as_matrix_block(X))
        addCol!(V, Vector{T}(x))
    end
    return V
end

function add_primal_basis!(estimator::PrimalDualErrorEstimator, X::AbstractVecOrMat)
    size(X, 1) == size(estimator.V, 1) || throw(DimensionMismatch("basis block has the wrong row dimension"))
    Xmat = _as_matrix_block(X)
    galerkin_add!(estimator.primal_rom, estimator.primal_model, Xmat, estimator.V)
    _append_basis_block!(estimator.V, Xmat)
    return estimator
end

function add_dual_basis!(estimator::PrimalDualErrorEstimator, X::AbstractVecOrMat)
    size(X, 1) == size(estimator.Vdu, 1) || throw(DimensionMismatch("basis block has the wrong row dimension"))
    Xmat = _as_matrix_block(X)
    galerkin_add!(estimator.dual_rom, estimator.dual_model, Xmat, estimator.Vdu)
    _append_basis_block!(estimator.Vdu, Xmat)
    return estimator
end

function add_residual_dual_basis!(estimator::FengBennerErrorEstimator, X::AbstractVecOrMat)
    size(X, 1) == size(estimator.Vrdu, 1) || throw(DimensionMismatch("basis block has the wrong row dimension"))
    Xmat = _as_matrix_block(X)
    galerkin_add!(estimator.rdu_rom, estimator.dual_model, Xmat, estimator.Vrdu)
    _append_basis_block!(estimator.Vrdu, Xmat)
    return estimator
end

function rebuild_estimator!(estimator::FengBennerErrorEstimator)
    estimator.primal_rom = galerkin_project(estimator.primal_model, estimator.V)
    estimator.dual_rom = galerkin_project(estimator.dual_model, estimator.Vdu)
    estimator.rdu_rom = galerkin_project(estimator.dual_model, estimator.Vrdu)
    return estimator
end

function update_estimator!(estimator::PrimalDualErrorEstimator, x::AbstractVector)
    return add_primal_basis!(estimator, x)
end

# -----------------------------------------------------------------------------
# Solves and residuals
# -----------------------------------------------------------------------------

function _form_operator_and_rhs!(model::LinearMatrixModel, p, i::Int)
    formArray!(model.Ap, model.A_alloc, p)
    formArray!(model.bps[i], model.b_alloc, p)
    return model.A_alloc, model.b_alloc
end

function _ensure_factorized!(model::LinearMatrixModel, p)
    if model.p_last != p
        formArray!(model.Ap, model.A_alloc, p)
        model.fA = factorize(model.A_alloc)
        model.p_last = p
    end
    return model.fA
end

function _solve_with_rhs!(model::LinearMatrixModel, p, rhs::AbstractVector)
    F = _ensure_factorized!(model, p)
    return F \ rhs
end

function primal_snapshot(estimator::PrimalDualErrorEstimator, sample; input::Int=1)
    p_concat = _transfer_parameter(sample)
    return estimator.primal_model(p_concat, input)
end

function dual_snapshot(estimator::PrimalDualErrorEstimator, sample; output::Int=1)
    p_concat = _transfer_parameter(sample)
    return estimator.dual_model(p_concat, output)
end

function _dual_residual_prepared(estimator::FengBennerErrorEstimator, p_concat, output::Int)
    z_du = estimator.dual_rom(p_concat, output)
    y_du = estimator.Vdu * z_du
    Qadj, cdual = _form_operator_and_rhs!(estimator.dual_model, p_concat, output)
    return cdual - Qadj * y_du
end

function residual_dual_snapshot(
    estimator::FengBennerErrorEstimator,
    sample;
    output::Int=1,
)
    p_concat = _transfer_parameter(sample)
    r_du = _dual_residual_prepared(estimator, p_concat, output)
    return _solve_with_rhs!(estimator.dual_model, p_concat, r_du)
end

function _estimate_components_prepared(
    estimator::FengBennerErrorEstimator,
    p_concat,
    input::Int,
    output::Int;
    z_primal=nothing,
)
    1 <= input <= length(estimator.primal_model.bps) || throw(BoundsError(estimator.primal_model.bps, input))
    1 <= output <= length(estimator.dual_model.bps) || throw(BoundsError(estimator.dual_model.bps, output))

    z_pr = isnothing(z_primal) ? estimator.primal_rom(p_concat, input) : z_primal
    length(z_pr) == size(estimator.V, 2) || throw(DimensionMismatch("x_r has length $(length(z_pr)); expected $(size(estimator.V, 2))"))
    x_pr = estimator.V * z_pr

    Q, b = _form_operator_and_rhs!(estimator.primal_model, p_concat, input)
    r_pr = b - Q * x_pr

    z_du = estimator.dual_rom(p_concat, output)
    y_du = estimator.Vdu * z_du
    Qadj, cdual = _form_operator_and_rhs!(estimator.dual_model, p_concat, output)
    r_du = cdual - Qadj * y_du

    rhs_rdu = estimator.Vrdu' * r_du
    z_rdu = _solve_with_rhs!(estimator.rdu_rom, p_concat, rhs_rdu)
    y_rdu = estimator.Vrdu * z_rdu

    residual_dual = abs(dot(y_rdu, r_pr))
    dual = abs(dot(y_du, r_pr))

    return FengBennerErrorComponents(
        Float64(residual_dual + dual),
        Float64(residual_dual),
        Float64(dual),
        input,
        output,
    )
end

function estimate_components(
    estimator::FengBennerErrorEstimator,
    sample;
    input::Int=1,
    output::Int=1,
)
    p_concat = _transfer_parameter(sample)
    return _estimate_components_prepared(estimator, p_concat, input, output)
end

function _sample_error_extrema(estimator::FengBennerErrorEstimator, sample)
    p_concat = _transfer_parameter(sample)
    best_total = FengBennerErrorComponents(-Inf, -Inf, -Inf, 0, 0)
    best_residual_dual = FengBennerErrorComponents(-Inf, -Inf, -Inf, 0, 0)

    for input in 1:length(estimator.primal_model.bps), output in 1:length(estimator.dual_model.bps)
        comp = _estimate_components_prepared(estimator, p_concat, input, output)
        if comp.total > best_total.total
            best_total = comp
        end
        if comp.residual_dual > best_residual_dual.residual_dual
            best_residual_dual = comp
        end
    end

    return best_total, best_residual_dual
end

function worst_estimate(estimator::FengBennerErrorEstimator, sample)
    return _sample_error_extrema(estimator, sample)[1]
end

function estimate(
    estimator::FengBennerErrorEstimator,
    sample;
    aggregate::Symbol=:max_entry,
    input::Int=1,
    output::Int=1,
)
    if aggregate == :max_entry
        return worst_estimate(estimator, sample).total
    elseif aggregate == :entry || aggregate == :first
        return estimate_components(estimator, sample; input=input, output=output).total
    elseif aggregate == :matrix
        p_concat = _transfer_parameter(sample)
        vals = zeros(Float64, length(estimator.dual_model.bps), length(estimator.primal_model.bps))
        for input in 1:length(estimator.primal_model.bps), output in 1:length(estimator.dual_model.bps)
            vals[output, input] = _estimate_components_prepared(estimator, p_concat, input, output).total
        end
        return vals
    else
        error("Unknown aggregate=$aggregate; choose :max_entry, :entry, :first, or :matrix")
    end
end

function estimate_all(estimator::FengBennerErrorEstimator, samples; kwargs...)
    return [estimate(estimator, sample; kwargs...) for sample in samples]
end

function approximate_error(
    estimator::FengBennerErrorEstimator,
    x_r::AbstractVector,
    sample,
    input::Int=1,
)
    p_concat = _transfer_parameter(sample)
    best = FengBennerErrorComponents(-Inf, -Inf, -Inf, input, 0)
    for output in eachindex(estimator.dual_model.bps)
        comp = _estimate_components_prepared(estimator, p_concat, input, output; z_primal=x_r)
        if comp.total > best.total
            best = comp
        end
    end
    return best.total
end
