"""
`sg_reductor <: SGReductor`

A struct for holding the parts of an SG reductor for 
a `StationaryModel` model. Can access the FOM through 
`sg_reductor.model`, the snapshot matrix through 
`sg_reductor.snapshots`, the pivoted QR object through 
`sg_reductor.decomp`, and the reduced basis through 
`sg_reductor.V`.
"""
struct SGReductor
    model::StationaryModel
    snapshots::AbstractMatrix
    decomp::Factorization
    V::AbstractMatrix
end

function Base.show(io::Core.IO, reductor::SGReductor)
    res  = "SG Reductor with $(size(reductor.snapshots)) snapshot matrix."
    println(io, res)
    print(io, "FOM: ")
    print(io, reductor.model)
end

"""
`sg_reductor = SGReductor(model, snapshots[; noise=-1])`

Forms an `SGReductor` object by computing the pivoted QR
decomposition of the matrix `snapshots`.
"""
function SGReductor(model::StationaryModel{NOUT}, snapshots::AbstractMatrix; noise=1) where NOUT
    if noise >= 1
        println("Forming column-pivoted QR decomposition of snapshots")
    end
    decomp = qr(snapshots, ColumnNorm())
    V = Matrix(decomp.Q)

    return SGReductor(model, snapshots, decomp, V)
end

"""
`sg_reductor = SGReductor(model, parameters[; noise=-1])`

Forms an `SGReductor` object by computing full order solutions
on each parameter in the vector `parameters` to form the snaphshot
matrix, and then calling `SGReductor(model, snapshots, noise=noise)`.
"""
function SGReductor(model::StationaryModel{NOUT}, parameters::AbstractVector; noise=1, progress=true) where NOUT
    p = parameters[1]
    if noise >= 1
        println("Forming snapshot matrix")
    end
    x = model(p)
    N = output_length(model)
    M = length(parameters)
    snapshots = Matrix{output_type(model)}(undef, N, M*NOUT)
    for (i,p) in (progress ? ProgressBar(enumerate(parameters)) : enumerate(parameters))
        for j in 1:NOUT
            idx = (i-1)*NOUT + j
            snapshots[:,idx] .= model(p, j)
        end
    end
    
    return SGReductor(model, snapshots, noise=noise)
end

"""
`form_rom(sg_reductor, r=-1)`

Calls `galerkin_project` on the FOM and returns
a ROM with RB of dimension `r`. If `r=-1`, uses
all available columns of `sg_reductor.V`.
"""
function form_rom(sg_reductor::SGReductor, r=-1)
    V = sg_reductor.V
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
    return view(V, 1:N, 1:r) * x_r
end