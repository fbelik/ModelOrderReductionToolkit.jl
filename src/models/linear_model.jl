"""
```
model = LinearModel(Ap::APArray, bp::APArray) <: StationaryModel{1}
model = LinearModel(Ap::APArray, b::AbstractVector) <: StationaryModel{1}
```

Struct for containing a parameterized linear model
`A(p) x = b(p)` or `A(p) x = b` with affine parameter 
dependence. Can form a solution for a new parameter value by 
calling it on a new parameter value `x = model(p)`.
"""
mutable struct LinearModel <: StationaryModel{1}
    Ap::APArray
    bp::APArray
    A_alloc::AbstractMatrix
    b_alloc::AbstractVector
end

function LinearModel(Ap::APArray, bp::APArray)
    TA = typeof(prod(zero.(eltype.(Ap.arrays))))
    A_alloc = Matrix{TA}(undef, size(Ap.arrays[1]))
    Tb = typeof(prod(zero.(eltype.(bp.arrays))))
    b_alloc = Vector{TA}(undef, size(bp.arrays[1]))
    return LinearModel(Ap, bp, A_alloc, b_alloc)
end

function LinearModel(Ap::APArray, b::AbstractVector)
    bp = APArray([b], (p,i) -> 1.0)
    return LinearModel(Ap, bp)
end

function Base.show(io::Core.IO, model::LinearModel)
    res  = "A(p) x(p) = b(p) with output length $(size(model.A_alloc, 2))"
    println(io, res)
    print(io, "A - ")
    println(io, model.Ap)
    print(io, "b - ")
    print(io, model.bp)
end

function (model::LinearModel)(p, i::Int=1)
    formArray!(model.Ap, model.A_alloc, p)
    formArray!(model.bp, model.b_alloc, p)
    return model.A_alloc \ model.b_alloc
end

function output_length(model::LinearModel)
    return size(model.A_alloc,2)
end

function output_type(model::LinearModel)
    T1 = eltype(model.A_alloc)
    T2 = eltype(model.b_alloc)
    return typeof(zero(T1) * zero(T2))
end

"""
`rom = galerkin_project(model::LinearModel, V[, W=V; r=-1])`

Perform (Petrov) Galerkin projection on a linear model
where the trial space is the first `r` columns of `V` and the test
space is the first `r` columns of `W`. If `r=-1`, uses all columns.
Returns a new `LinearModel`.

`W' * A(p) * V * x_r = W' * b(p), `V * x_r ≈ x = A(p)^(-1) b(p)`
"""
function galerkin_project(model::LinearModel, V::AbstractMatrix, W::AbstractMatrix=V; r=-1)
    N, n = size(V)
    if 0 < r && r < min(N, n)
        V = view(V, 1:N, 1:r)
        W = view(W, 1:N, 1:r)
    end

    reduced_Ais = [VectorOfVectors(W' * Ai * V) for Ai in model.Ap.arrays]
    reduced_bis = [W' * bi for bi in model.bp.arrays]

    Apr = APArray(reduced_Ais, model.Ap.makeθi)
    bpr = APArray(reduced_bis, model.bp.makeθi)

    return LinearModel(Apr, bpr)
end

"""
`galerkin_add!(rom::LinearModel, fom::LinearModel, v, Vold[, w=v, Wold=Vold; r=-1])`

Assuming that `rom = galerkin_project(model, Vold, Wold)`,
updates `rom` such that if `V = [Vold v]` and `W = [Wold w]`, then
`rom = galerkin_project(model, V, W)`.
"""
function galerkin_add!(rom::LinearModel, fom::LinearModel, v::AbstractVecOrMat, Vold::AbstractMatrix, w::AbstractVecOrMat=v, Wold::AbstractMatrix=Vold)
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
    # Update rom.bp and rom.b_alloc
    for (b,br) in zip(fom.bp.arrays, rom.bp.arrays)
        push!(br, (w' * b)...)
    end
    push!(rom.b_alloc, (w' * fom.b_alloc)...)
    nothing
end