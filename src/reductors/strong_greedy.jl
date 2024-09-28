"""
`sg_reductor <: SGReductor`

A struct for holding the parts of an SG reductor for 
a `StationaryModel` model. Can access the FOM through 
`sg_reductor.model`, the snapshot matrix through 
`sg_reductor.snapshots`, the pivot order through 
`sg_reductor.p`, and the reduced basis through 
`sg_reductor.V`.
"""
struct SGReductor{NOUT}
    model::StationaryModel{NOUT}
    snapshots::VectorOfVectors
    p::Vector{Int}
    parameters::Set
    V::VectorOfVectors
end

function Base.show(io::Core.IO, reductor::SGReductor)
    res  = "SG Reductor with $(size(reductor.snapshots)) snapshot matrix"
    println(io, res)
    print(io, "FOM: ")
    println(io, reductor.model)
    print(io, "Increase RB dimension with add_to_rb!(reductor, params)")
end

"""
`sg_reductor = SGReductor(model)`

Forms a `SGReductor` object for `model <: StationaryModel`.
"""
function SGReductor(model::StationaryModel) 
    T = output_type(model)
    N = output_length(model)
    snapshots = VectorOfVectors(N, 0, T)
    p = Int[]
    parameters = Set()
    V = VectorOfVectors(N, 0, T)
    return SGReductor(model, snapshots, p, parameters, V)
end

"""
`add_to_rb!(sg_reductor, snapshots[; noise=1])`

Directly updates `sg_reductor` with new snapshots given in
the columns of the matrix `snapshots`.
"""
function add_to_rb!(sg_reductor::SGReductor, snapshots::AbstractMatrix; noise=1)
    @assert size(snapshots, 1) == size(sg_reductor.snapshots, 1)
    for x in eachcol(snapshots)
        addCol!(sg_reductor.snapshots)
        if size(sg_reductor.V, 2) < output_length(sg_reductor.model)
            addCol!(sg_reductor.V)
            push!(sg_reductor.p, 0)
        end
        sg_reductor.snapshots[:,end] .= x
    end
    if noise >= 1
        println("Forming column-pivoted QR of snapshot matrix")
    end
    decomp = qr(sg_reductor.snapshots, ColumnNorm())
    sg_reductor.V .= decomp.Q[:, 1:min(size(sg_reductor.snapshots, 2),output_length(sg_reductor.model))]
    sg_reductor.p .= decomp.p[eachindex(sg_reductor.p)]
    nothing
end


"""
`add_to_rb!(sg_reductor, parameters[; noise=1, progress=true])`

Loops through the vector of `parameters`, forms their full order solutions,
adds them to `sg_reductor.snapshots`, and then updates the reduced basis
in `sg_reductor.V`.
"""
function add_to_rb!(sg_reductor::SGReductor{NOUT}, parameters::AbstractVector; noise=1, progress=true) where NOUT
    if noise >= 1
        println("Adding to RB by forming full order solutions")
    end
    for (i,p) in (progress ? ProgressBar(enumerate(parameters)) : enumerate(parameters))
        if p in sg_reductor.parameters
            continue
        end
        for j in 1:NOUT
            addCol!(sg_reductor.snapshots)
            if size(sg_reductor.V, 2) < output_length(sg_reductor.model)
                addCol!(sg_reductor.V)
                push!(sg_reductor.p, 0)
            end
            sg_reductor.snapshots[:,end] .= sg_reductor.model(p, j)
        end
        push!(sg_reductor.parameters, p)
    end
    if noise >= 1
        println("Forming column-pivoted QR of snapshot matrix")
    end
    decomp = qr(sg_reductor.snapshots, ColumnNorm())
    sg_reductor.V .= decomp.Q[:, 1:min(size(sg_reductor.snapshots, 2),output_length(sg_reductor.model))]
    sg_reductor.p .= decomp.p[eachindex(sg_reductor.p)]
    nothing
end

"""
`form_rom(sg_reductor, r=-1)`

Calls `galerkin_project` on the FOM and returns
a ROM with RB of dimension `r`. If `r=-1`, uses
all available columns of `sg_reductor.V`.
"""
function form_rom(sg_reductor::SGReductor, r=-1)
    V = Matrix(sg_reductor.V)
    return galerkin_project(sg_reductor.model, V, r=r)
end

"""
`lift(sg_reductor, x_r)`

Given a vector solution `x_r` to a ROM formed by the
`sg_reductor`, which is of smaller dimension than outputs
of the FOM, lifts the solution to the same dimension of
the FOM. 
"""
function lift(reductor::SGReductor, x_r::AbstractVector)
    r = length(x_r)
    V = reductor.V
    N, M = size(V)
    return view(Matrix(V), 1:N, 1:r) * x_r
end