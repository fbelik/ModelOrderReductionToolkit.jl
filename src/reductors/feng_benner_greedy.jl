mutable struct FengBennerGreedyReductor
    model::LTIModel
    samples::AbstractVector
    estimator::FengBennerErrorEstimator
    rom::LTIModel
    selected_samples::Vector{Any}
    selected_rdu_samples::Vector{Any}
    approx_errors::Vector{Float64}
    residual_dual_errors::Vector{Float64}
    truth_errors::Vector{Float64}
    force_rb_real::Bool
    directions::Symbol
    include_dual_in_residual_dual::Bool
end

function Base.show(io::IO, red::FengBennerGreedyReductor)
    print(io,
        "Transfer greedy reductor with ",
        "dim(V)=", size(red.estimator.V, 2), ", ",
        "dim(Vdu)=", size(red.estimator.Vdu, 2), ", ",
        "dim(Vrdu)=", size(red.estimator.Vrdu, 2),
    )
end

"""
    FengBennerGreedyReductor(model, samples; kwargs...)

Create a transfer-function greedy reductor for an `LTIModel`. The reductor
builds the primal, dual, and residual-dual spaces used by
`FengBennerErrorEstimator`.

Keyword arguments:

  * `force_rb_real=false`: split complex snapshots into real/imaginary parts.
  * `logfreq=false`: treat the first sample coordinate as `log10(omega)`.
  * `directions=:worst_pair`: enrich only the current worst input/output entry.
    Use `:all` to enrich all input and output directions at each selected sample.
  * `initialize=true`: seed `V`, `Vdu`, and `Vrdu` before the first greedy step.
  * `include_dual_in_residual_dual=true`: keep the dual space contained in
    the residual-dual space, then augment it with residual-dual snapshots.
"""
function FengBennerGreedyReductor(
    model::LTIModel,
    samples::AbstractVector,
    primal_model::Union{Nothing,LinearMatrixModel}=nothing,
    dual_model::Union{Nothing,LinearMatrixModel}=nothing;
    force_rb_real::Bool=false,
    logfreq::Bool=false,
    directions::Symbol=:worst_pair,
    initialize::Bool=true,
    start_sample=nothing,
    start_rdu_sample=nothing,
    zero_tol::Real=1e-14,
    include_dual_in_residual_dual::Bool=true,
)
    length(samples) > 0 || error("samples must be nonempty")
    directions in (:worst_pair, :all) || error("directions must be :worst_pair or :all")

    estimator = FengBennerErrorEstimator(model, primal_model, dual_model; force_rb_real, logfreq)
    red = FengBennerGreedyReductor(
        model,
        samples,
        estimator,
        galerkin_project(model, estimator.V),
        Any[],
        Any[],
        Float64[],
        Float64[],
        Float64[],
        force_rb_real,
        directions,
        include_dual_in_residual_dual,
    )

    if initialize
        s0 = isnothing(start_sample) ? first(samples) : start_sample
        sr = isnothing(start_rdu_sample) ? last(samples) : start_rdu_sample
        _transfer_add_primal_dual_snapshots!(red, s0; input=1, output=1, zero_tol=zero_tol)
        _transfer_add_residual_dual_snapshots!(red, sr; output=1, zero_tol=zero_tol)
        _update_transfer_rom!(red)
        push!(red.selected_samples, _transfer_sample_copy(s0))
        push!(red.selected_rdu_samples, _transfer_sample_copy(sr))
    end

    return red
end

function _transfer_sample_copy(sample)
    sample isa AbstractArray && return copy(sample)
    if sample isa Tuple && length(sample) == 2 && sample[2] isa AbstractArray
        return (sample[1], copy(sample[2]))
    end
    return sample
end

function _transfer_direction_indices(n::Int, directions::Symbol, selected::Int)
    1 <= selected <= n || throw(BoundsError(1:n, selected))
    return directions == :all ? (1:n) : (selected:selected)
end

function _transfer_add_snapshot_to_space!(
    add_snapshot!::Function,
    V::VectorOfVectors{T},
    x;
    force_real::Bool,
    zero_tol::Real,
) where T
    added = 0
    if force_real && eltype(x) <: Complex
        for part in (real.(x), imag.(x))
            u = Vector{T}(part)
            nrm = orthonormalize_mgs2!(u, V)
            if nrm >= zero_tol
                add_snapshot!(u)
                added += 1
            end
        end
    else
        u = Vector{T}(x)
        nrm = orthonormalize_mgs2!(u, V)
        if nrm >= zero_tol
            add_snapshot!(u)
            added += 1
        end
    end
    return added
end

function _transfer_add_primal_snapshot!(
    red::FengBennerGreedyReductor,
    sample;
    input::Int,
    zero_tol::Real,
)
    x = primal_snapshot(red.estimator, sample; input=input)
    return _transfer_add_snapshot_to_space!(
        x -> add_primal_basis!(red.estimator, x),
        red.estimator.V,
        x;
        force_real=red.force_rb_real,
        zero_tol=zero_tol,
    )
end

function _transfer_add_dual_snapshot!(
    red::FengBennerGreedyReductor,
    sample;
    output::Int,
    zero_tol::Real,
)
    x = dual_snapshot(red.estimator, sample; output=output)
    added = _transfer_add_snapshot_to_space!(
        x -> add_dual_basis!(red.estimator, x),
        red.estimator.Vdu,
        x;
        force_real=red.force_rb_real,
        zero_tol=zero_tol,
    )
    if red.include_dual_in_residual_dual
        added += _transfer_add_snapshot_to_space!(
            x -> add_residual_dual_basis!(red.estimator, x),
            red.estimator.Vrdu,
            x;
            force_real=red.force_rb_real,
            zero_tol=zero_tol,
        )
    end
    return added
end

function _transfer_add_residual_dual_snapshot!(
    red::FengBennerGreedyReductor,
    sample;
    output::Int,
    zero_tol::Real,
)
    x = residual_dual_snapshot(red.estimator, sample; output=output)
    return _transfer_add_snapshot_to_space!(
        x -> add_residual_dual_basis!(red.estimator, x),
        red.estimator.Vrdu,
        x;
        force_real=red.force_rb_real,
        zero_tol=zero_tol,
    )
end

function _transfer_add_primal_dual_snapshots!(
    red::FengBennerGreedyReductor,
    sample;
    input::Int,
    output::Int,
    zero_tol::Real,
)
    inputs = _transfer_direction_indices(length(red.estimator.primal_model.bps), red.directions, input)
    outputs = _transfer_direction_indices(length(red.estimator.dual_model.bps), red.directions, output)

    added = 0
    for j in inputs
        added += _transfer_add_primal_snapshot!(red, sample; input=j, zero_tol=zero_tol)
    end
    for i in outputs
        added += _transfer_add_dual_snapshot!(red, sample; output=i, zero_tol=zero_tol)
    end
    return added
end

function _transfer_add_residual_dual_snapshots!(
    red::FengBennerGreedyReductor,
    sample;
    output::Int,
    zero_tol::Real,
)
    outputs = _transfer_direction_indices(length(red.estimator.dual_model.bps), red.directions, output)
    added = 0
    for i in outputs
        added += _transfer_add_residual_dual_snapshot!(red, sample; output=i, zero_tol=zero_tol)
    end
    return added
end

function _update_transfer_rom!(red::FengBennerGreedyReductor)
    red.rom = galerkin_project(red.model, red.estimator.V)
    return red.rom
end

function _transfer_sample_omega_and_parameter(red::FengBennerGreedyReductor, sample)
    p = _transfer_parameter(sample)
    omega = red.estimator.logfreq ? exp10(p[1]) : p[1]
    phys = length(p) == 1 ? nothing : view(p, 2:length(p))
    return omega, phys
end

function true_transfer_error(
    red::FengBennerGreedyReductor,
    sample;
    aggregate::Symbol=:max_entry,
)
    omega, p = _transfer_sample_omega_and_parameter(red, sample)
    H = isnothing(p) ? bode(red.model, omega; first=false) : bode(red.model, omega, p; first=false)
    Hr = isnothing(p) ? bode(red.rom, omega; first=false) : bode(red.rom, omega, p; first=false)
    E = H .- Hr

    if aggregate == :max_entry
        return Float64(maximum(abs.(E)))
    elseif aggregate == :frobenius
        return Float64(norm(E))
    elseif aggregate == :first
        return Float64(abs(E[1, 1]))
    else
        error("Unknown aggregate=$aggregate; choose :max_entry, :frobenius, or :first")
    end
end

function _transfer_greedy_step!(
    red::FengBennerGreedyReductor;
    eps::Real,
    noise::Int,
    progress::Bool,
    zero_tol::Real,
)
    best_total = FengBennerErrorComponents(-Inf, -Inf, -Inf, 0, 0)
    best_rdu = FengBennerErrorComponents(-Inf, -Inf, -Inf, 0, 0)
    best_total_sample = nothing
    best_rdu_sample = nothing

    sample_iter = progress ? ProgressBar(red.samples) : red.samples
    for sample in sample_iter
        sample_best_total, sample_best_rdu = _sample_error_extrema(red.estimator, sample)
        if sample_best_total.total > best_total.total
            best_total = sample_best_total
            best_total_sample = sample
        end
        if sample_best_rdu.residual_dual > best_rdu.residual_dual
            best_rdu = sample_best_rdu
            best_rdu_sample = sample
        end
    end

    isnothing(best_total_sample) && return false

    if best_total.total < eps
        noise >= 1 && @printf(
            "maximum estimator %.4e < eps %.4e; not appending snapshots\n",
            best_total.total,
            eps,
        )
        return false
    end

    truth_error = true_transfer_error(red, best_total_sample; aggregate=:max_entry)
    push!(red.approx_errors, best_total.total)
    push!(red.residual_dual_errors, best_rdu.residual_dual)
    push!(red.truth_errors, truth_error)
    push!(red.selected_samples, _transfer_sample_copy(best_total_sample))
    push!(red.selected_rdu_samples, _transfer_sample_copy(best_rdu_sample))

    old_dim = size(red.estimator.V, 2) + size(red.estimator.Vdu, 2) + size(red.estimator.Vrdu, 2)
    _transfer_add_primal_dual_snapshots!(
        red,
        best_total_sample;
        input=best_total.input,
        output=best_total.output,
        zero_tol=zero_tol,
    )
    _transfer_add_residual_dual_snapshots!(
        red,
        best_rdu_sample;
        output=best_rdu.output,
        zero_tol=zero_tol,
    )
    _update_transfer_rom!(red)

    new_dim = size(red.estimator.V, 2) + size(red.estimator.Vdu, 2) + size(red.estimator.Vrdu, 2)
    added = new_dim > old_dim

    if noise >= 1
        @printf(
            "estimator %.4e, highest residual dual %.4e, truth %.4e, dims=(%d,%d,%d)\n",
            best_total.total,
            best_rdu.residual_dual,
            truth_error,
            size(red.estimator.V, 2),
            size(red.estimator.Vdu, 2),
            size(red.estimator.Vrdu, 2),
        )
    end
    !added && noise >= 1 && println("No linearly independent snapshots were added; stopping")

    return added
end

function add_to_rb!(
    red::FengBennerGreedyReductor;
    maxiter::Int=1,
    eps::Real=0.0,
    noise::Int=0,
    progress::Bool=false,
    zero_tol::Real=1e-14,
)
    all_added = true
    for _ in 1:maxiter
        added = _transfer_greedy_step!(
            red;
            eps=eps,
            noise=noise,
            progress=progress,
            zero_tol=zero_tol,
        )
        if !added
            all_added = false
            break
        end
    end
    return all_added
end

function add_to_rb!(red::FengBennerGreedyReductor, samples::AbstractVector; kwargs...)
    old_samples = red.samples
    red.samples = samples
    try
        return add_to_rb!(red; kwargs...)
    finally
        red.samples = old_samples
    end
end

function form_rom(red::FengBennerGreedyReductor, r::Int=-1)
    if r == -1 || r == size(red.estimator.V, 2)
        return red.rom
    elseif 0 <= r <= size(red.estimator.V, 2)
        return galerkin_project(red.model, Matrix(red.estimator.V)[:, 1:r])
    else
        error("r=$r must satisfy 0 <= r <= $(size(red.estimator.V, 2))")
    end
end

function get_rom(red::FengBennerGreedyReductor)
    return red.rom
end

function lift(red::FengBennerGreedyReductor, x_r::AbstractArray)
    r = size(x_r, 1)
    return red.estimator.V[:, 1:r] * x_r
end
