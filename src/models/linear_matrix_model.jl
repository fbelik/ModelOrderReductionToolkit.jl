"""
```
model = LinearMatrixModel(Ap::APArray, bps::AbstractVector{Union{APArray, <:AbstractVector}})
model = LinearMatrixModel(Ap::APArray, Bp::APArray)
model = LinearMatrixModel(models::AbstractVector{LinearModel})
```

Struct for containing a parameterized linear model
`A(p) X = B(p)`, which is stored as a single `LinearModel`, with 
affine parameter dependence by concatenating the columns of `B(p)`
and making the parameter a tuple where `p[1]` denotes the column
of `X` to be solved for, and `p[2]` denotes the standard parameter.
Ex. `X[:,i] = model((i, p))`.
"""
mutable struct LinearMatrixModel{NOUT} <: StationaryModel{NOUT}
    Ap::APArray
    bps::AbstractVector{APArray}
    A_alloc::AbstractMatrix
    b_alloc::AbstractVector
    B_alloc::VectorOfVectors
end

function LinearMatrixModel(Ap::APArray, bps::AbstractVector{APArray})
    TA = typeof(prod(zero.(eltype.(Ap.arrays))))
    A_alloc = Matrix{TA}(undef, size(Ap.arrays[1]))
    Tb = typeof(prod([prod(zero.(eltype.(bp.arrays))) for bp in bps]))
    b_alloc = Vector{Tb}(undef, length(bps[1].arrays[1]))
    B_alloc = VectorOfVectors(Matrix{Tb}(undef, length(b_alloc), length(bps)))
    return LinearMatrixModel{length(bps)}(Ap, bps, A_alloc, b_alloc, B_alloc)
end

function LinearMatrixModel(Ap::APArray, bps::AbstractVector{Union{APArray,<:AbstractVector}})
    # Convert each bp to an APArray if not already
    bps = copy(bps)
    for i in eachindex(bps)
        if isa(bps[i], AbstractVector)
            bps[i] = APArray([bps[i]], (p,i) -> 1.0)
        end
    end
    return LinearMatrixModel(Ap, bps)
end

function LinearMatrixModel(Ap::APArray, Bp::APArray)
    ncols = size(Bp.arrays[1],2)
    bps = [APArray([b[:,i] for b in Bp.arrays], Bp.makeθi) for i in 1:ncols]
    return LinearMatrixModel(Ap, bps)
end

function LinearMatrixModel(Ap::APArray, B::AbstractMatrix)
    bps = [APArray([b], (p,i) -> 1.0) for b in eachcol(B)]
    return LinearMatrixModel(Ap, bps)
end

function Base.show(io::Core.IO, model::LinearMatrixModel{NOUT}) where NOUT
    res  = "A(p) xᵢ(p) = bᵢ(p) for i=1:$NOUT with output dimension $(size(model.b_alloc)), $(length(model.Ap.arrays)) LHS affine terms, and $([length(bp.arrays) for bp in model.bps]) RHS affine terms"
    print(io, res)
end

function (model::LinearMatrixModel{NOUT})(p, i::Int=1) where NOUT
    formArray!(model.Ap, model.A_alloc, p)
    if i == 0
        for j in 1:NOUT
            formArray!(model.bps[j], view(model.B_alloc, :, j), p)
        end
        return model.A_alloc \ model.B_alloc
    end
    formArray!(model.bps[i], model.b_alloc, p)
    return model.A_alloc \ model.b_alloc
end

function output_length(model::LinearMatrixModel)
    return size(model.A_alloc,2)
end

function output_type(model::LinearMatrixModel)
    T1 = eltype(model.A_alloc)
    T2 = eltype(model.b_alloc)
    if T1 == T2
        return T1
    else
        return typeof(zero(T1) * zero(T2))
    end
end

"""
`rom = galerkin_project(model::LinearMatrixModel, V[, W=V; r=-1])`

Perform (Petrov) Galerkin projection on each linear model in `model.models`.
"""
function galerkin_project(model::LinearMatrixModel{NOUT}, V::AbstractMatrix, W::AbstractMatrix=V; r=-1) where NOUT
    N, n = size(V)
    if 0 < r && r < min(N, n)
        V = view(V, 1:N, 1:r)
        W = view(W, 1:N, 1:r)
    end

    reduced_Ais = [VectorOfVectors(W' * Ai * V) for Ai in model.Ap.arrays]
    Apr = APArray(reduced_Ais, model.Ap.makeθi)

    bprs = APArray[]
    for i in 1:NOUT
        reduced_bis = [W' * bi for bi in model.bps[i].arrays]
        bpr = APArray(reduced_bis, model.bps[i].makeθi)
        push!(bprs, bpr)
    end

    return LinearMatrixModel(Apr, bprs)
end

"""
`galerkin_add!(rom::LinearMatrixModel, fom::LinearMatrixModel, v, Vold[, w=v, Wold=Vold; r=-1])`

Assuming that `rom = galerkin_project(fom, Vold, Wold)`,
updates `rom` such that if `V = [Vold v]` and `W = [Wold w]`, then
`rom = galerkin_project(fom, V, W)`.
"""
function galerkin_add!(rom::LinearMatrixModel{NOUT}, fom::LinearMatrixModel, v::AbstractVecOrMat, Vold::AbstractMatrix, w::AbstractVecOrMat=v, Wold::AbstractMatrix=Vold) where NOUT
    r = size(rom.A_alloc, 1)
    radd = size(v, 2)
    N = size(Vold, 1)
    # Update rom.Ap and rom.A_alloc
    if all(isa.(rom.Ap.arrays, VectorOfVectors))
        for (A,Ar) in zip(fom.Ap.arrays, rom.Ap.arrays)
            for i in 1:radd
                addCol!(Ar)
            end
            for i in 1:r
                Ar[i:i, (r+1):(r+radd)] .= view(Wold, 1:N, i)' * (A * v)
            end
            for i in 1:radd
                addRow!(Ar)
            end
            for i in 1:r
                Ar[(r+1):(r+radd), i:i] .= (w' * A) * view(Vold, 1:N, i)
            end
            Ar[(r+1):(r+radd),(r+1):(r+radd)] .= w' * (A * v)
        end
        rom.A_alloc = similar(rom.Ap.arrays[1])
    else
        # Reallocate arrays 
        Ais_new = [VectorOfVectors(Matrix{eltype(Ai)}(undef, r+radd, r+radd)) for Ai in rom.Ap.arrays]
        for j in eachindex(rom.Ap.arrays)
            Ais_new[j][1:r, 1:r] .= rom.Ap.arrays[j]
            for i in 1:r
                Ais_new[j][i:i, (r+1):(r+radd)] .= view(Wold, 1:N, i)' * (fom.Ap.arrays[j] * v)
            end
            for i in 1:r
                Ais_new[j][(r+1):(r+radd), i:i] .= (w' * fom.Ap.arrays[j]) * view(Vold, 1:N, i)
            end
            Ais_new[j][(r+1):(r+radd),(r+1):(r+radd)] .= w' * (fom.Ap.arrays[j] * v)
        end
        rom.Ap = APArray(Ais_new, rom.Ap.makeθi)
        rom.A_alloc = similar(rom.Ap.arrays[1])
    end
    # Update rom.bps and rom.b_alloc
    for i in 1:NOUT
        for (b,br) in zip(fom.bps[i].arrays, rom.bps[i].arrays)
            push!(br, (w' * b)...)
        end
    end
    push!(rom.b_alloc, (w' * fom.b_alloc)...)
    # Update rom.B_alloc
    if isa(rom.B_alloc, VectorOfVectors)
        for i in 1:size(w, 2)
            addRow!(rom.B_alloc)
        end
    else
        rom.B_alloc = VectorOfVectors(Matrix{eltype(rom.B_alloc)}(undef, length(rom.b_alloc), length(rom.bps)))
    end
    nothing
end