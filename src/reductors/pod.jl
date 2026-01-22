"""
`pod_reductor <: PODReductor`

A struct for holding the parts of a POD reductor for
a `StationaryModel` model. Can access the FOM through 
`pod_reductor.model`, the snapshot matrix through 
`pod_reductor.snapshots`, the singular values through 
`pod_reductor.S`, and the reduced basis 
(left singular vectors) through `pod_reductor.V`.
"""
struct PODReductor{NOUT}
    model::StationaryModel{NOUT}
    snapshots::VectorOfVectors
    parameters::Set
    S::Vector{Float64}
    V::VectorOfVectors
end

"""
`pod_reductor = PODReductor(model[; force_rb_real=false])`

Forms a `PODReductor` object for `model <: StationaryModel`. Forces
the RB to be real if `force_rb_real==true`.
"""
function PODReductor(model::StationaryModel{NOUT}; force_rb_real=false) where NOUT
    T = output_type(model)
    N = output_length(model)
    snapshots = VectorOfVectors(N, 0, T)
    parameters = Set()
    S = Float64[]
    V = VectorOfVectors(N, 0, force_rb_real ? real(T) : T)
    return PODReductor{NOUT}(model, snapshots, parameters, S, V)
end

function Base.show(io::Core.IO, reductor::PODReductor)
    res  = "POD Reductor with $(size(reductor.snapshots)) snapshot matrix"
    println(io, res)
    print(io, "FOM: ")
    println(io, reductor.model)
    print(io, "Increase RB dimension with add_to_rb!(reductor, params)")
end

"""
`add_to_rb!(pod_reductor, snapshots[; noise=1])`

Directly updates `pod_reductor` with new snapshots given in
the columns of the matrix `snapshots`.
"""
function add_to_rb!(pod_reductor::PODReductor, snapshots::AbstractMatrix; noise=0)
    @assert size(snapshots, 1) == size(pod_reductor.snapshots, 1)
    forced_real = eltype(pod_reductor.snapshots) <: Complex && eltype(pod_reductor.V) <: Real
    for x in eachcol(snapshots)
        addCol!(pod_reductor.snapshots)
        if size(pod_reductor.V, 2) < output_length(pod_reductor.model)
            addCol!(pod_reductor.V)
            push!(pod_reductor.S, 0.0)
            if size(pod_reductor.V, 2) < output_length(pod_reductor.model) && forced_real
                addCol!(pod_reductor.V)
                push!(pod_reductor.S, 0.0)
            end
        end
        pod_reductor.snapshots[:,end] .= x
    end
    if noise >= 1
        println("Forming SVD of snapshot matrix")
    end
    if forced_real
        real_snapshots = hcat(real.(pod_reductor.snapshots), imag.(pod_reductor.snapshots))
        U, S, _ = svd(real_snapshots)
        pod_reductor.S .= S
        pod_reductor.V .= U
    else
        U, S, _ = svd(pod_reductor.snapshots)
        pod_reductor.S .= S
        pod_reductor.V .= U
    end
    nothing
end

"""
`add_to_rb!(pod_reductor, parameters[; noise=0, progress=false])`

Loops through the vector of `parameters`, forms their full order solutions,
adds them to `pod_reductor.snapshots`, and then updates the singular values 
and singular vectors in `pod_reductor.S` and `pod_reductor.V`.
"""
function add_to_rb!(pod_reductor::PODReductor{NOUT}, parameters::AbstractVector; noise=0, progress=false) where NOUT
    if noise >= 1
        println("Adding to RB by forming full order solutions")
    end
    forced_real = eltype(pod_reductor.snapshots) <: Complex && eltype(pod_reductor.V) <: Real
    for p in (progress ? ProgressBar(parameters) : parameters)
        if p in pod_reductor.parameters
            continue
        end
        for j in 1:NOUT
            addCol!(pod_reductor.snapshots)
            if size(pod_reductor.V, 2) < output_length(pod_reductor.model)
                addCol!(pod_reductor.V)
                push!(pod_reductor.S, 0.0)
                if size(pod_reductor.V, 2) < output_length(pod_reductor.model) && forced_real
                    addCol!(pod_reductor.V)
                    push!(pod_reductor.S, 0.0)
                end
            end
            pod_reductor.snapshots[:,end] .= pod_reductor.model(p, j)
        end
        push!(pod_reductor.parameters, p)
    end
    if noise >= 1
        println("Forming SVD of snapshot matrix")
    end
    if forced_real
        U, S, _ = svd(hcat(real.(pod_reductor.snapshots), imag.(pod_reductor.snapshots)))
        pod_reductor.S .= S
        pod_reductor.V .= U
    else
        U, S, _ = svd(pod_reductor.snapshots)
        pod_reductor.S .= S
        pod_reductor.V .= U
    end
    nothing
end

"""
`form_rom(pod_reductor, r=-1)`

Pulls the first `r` left singular vectors from 
`pod_reductor.V`, and then Galerkin projects
`pod_reductor.model` onto that basis, and returns
the resulting ROM. If `r=-1`, uses largest size
possible.
"""
function form_rom(pod_reductor::PODReductor, r=-1)
    V = Matrix(pod_reductor.V)
    return galerkin_project(pod_reductor.model, V, r=r)
end

"""
`lift(pod_reductor, x_r)`

Given a solution array `x_r` to a ROM formed by the
`pod_reductor` lifts the solution(s) to the same dimension of
the FOM.  
"""
function lift(pod_reductor::PODReductor, x_r::AbstractArray)
    r = size(x_r,1)
    V = pod_reductor.V
    return V[:, 1:r] * x_r
end