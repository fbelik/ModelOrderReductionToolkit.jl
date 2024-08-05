"""
`pod_reductor <: PODReductor`

A struct for holding the parts of a POD reductor for
a `StationaryModel` model. Can access the FOM through 
`pod_reductor.model`, the snapshot matrix through 
`pod_reductor.snapshots`, the SVD object through 
`pod_reductor.decomp`, and the reduced basis 
(left singular vectors) through `pod_reductor.V`.
"""
struct PODReductor
    model::StationaryModel
    snapshots::AbstractMatrix
    decomp::Factorization
    V::AbstractMatrix
end

"""
`pod_reductor = PODReductor(model, snapshots[; noise=-1])`

Forms a `PODReductor` object by computing the SVD of the matrix
`snapshots`.
"""
function PODReductor(model::StationaryModel, snapshots::AbstractMatrix; noise=1)
    if noise >= 1
        println("Forming PCA decomposition of snapshots")
    end
    decomp = svd(snapshots)
    V = decomp.U

    return PODReductor(model, snapshots, decomp, V)
end

function Base.show(io::Core.IO, reductor::PODReductor)
    res  = "POD Reductor with $(size(reductor.snapshots)) snapshot matrix."
    println(io, res)
    print(io, "FOM: ")
    print(io, reductor.model)
end

"""
`pod_reductor = PODReductor(model, parameters[; noise=-1])`

Forms a `PODReductor` object by computing full order solutions
on each parameter in the vector `parameters` to form the snaphshot
matrix, and then calling `PODReductor(model, snapshots, noise=noise)`.
"""
function PODReductor(model::StationaryModel{NOUT}, parameters::AbstractVector; noise=1, progress=true) where NOUT
    p = parameters[1]
    if noise >= 1
        println("Forming snapshot matrix")
    end
    N = output_length(model)
    M = length(parameters)
    snapshots = Matrix{output_type(model)}(undef, N, M * NOUT)
    for (i,p) in (progress ? ProgressBar(enumerate(parameters)) : enumerate(parameters))
        for j in 1:NOUT
            idx = (i-1)*NOUT + j
            snapshots[:,idx] .= model(p, j)
        end
    end
    
    return PODReductor(model, snapshots, noise=noise)
end

"""
`form_rom(pod_reductor, r=-1)`

Pulls the first `r` left singular vectors from 
`pod_reductor.decomp`, and then Galerkin projects
`pod_reductor.model` onto that basis, and returns
the resulting ROM.
"""
function form_rom(pod_reductor::PODReductor, r=-1)
    V = pod_reductor.V
    return galerkin_project(pod_reductor.model, V, r=r)
end

"""
`lift(pod_reductor, x_r)`

Given a vector solution `x_r` to a ROM formed by the
`pod_reductor`, which is of smaller dimension than outputs
of the FOM, lifts the solution to the same dimension of
the FOM. 
"""
function lift(pod_reductor::PODReductor, x_r::AbstractVector)
    r = length(x_r)
    V = pod_reductor.V
    N, M = size(V)
    return view(V, 1:N, 1:r) * x_r
end