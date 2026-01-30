# import Base: +
import Base: size
import Base: getindex
import Base: setindex!
import Base: eachcol

"""
`VectorOfVectors{T} <: AbstractMatrix{T}`

Type for defining matrices as vectors of vectors for 
quick insertion of rows and columns. Stores the columns as
vectors in `vecs` and its size in `size`.

Construct an empty `VOV` of dimensions `(nrows × ncols)` where 
one of `nrows` or `ncols` must be zero:

`VectorOfVectors(nrows=0, ncols=0, T=Float64)`

Construct a `VOV` from a Julia vector of vectors:

`VectorOfVectors(vecs::AbstractVector{<:AbstractVector{T}})`

Construct a `VOV` from a matrix:

`VectorOfVectors(M::AbstractMatrix{T})`
"""
struct VectorOfVectors{T} <: AbstractMatrix{T}
    vecs::AbstractVector{<:AbstractVector{T}}
    size::MVector{2,Int}
    VectorOfVectors(nrows::Int=0, ncols::Int=0, T=Float64) = begin
        @assert (nrows==0 || ncols==0)
        new{T}(Vector{T}[], @MVector [nrows,ncols])
    end
    VectorOfVectors(vecs::AbstractVector{<:AbstractVector{T}}) where T = begin
        ncols = length(vecs)
        if ncols == 0
            nrows = 0
            return new{T}(Vector{T}[], @MVector [0,0])
        end
        nrows = length(vecs[1])
        for vec in vecs[2:end]
            if length(vec[i]) != nrows
                error("Each column must have the same number of elements")
            end
        end
        new{T}(vecs, @MVector [nrows,ncols])
    end
    VectorOfVectors(M::AbstractMatrix{T}) where T = begin
        nrows, ncols = size(M)
        vecs = [copy(col) for col in eachcol(M)]
        new{T}(vecs, @MVector [nrows,ncols])
    end
end

VOV = VectorOfVectors

@doc (@doc VectorOfVectors) VOV

function size(vov::VOV)
    return Tuple(vov.size)
end

function getindex(vov::VOV, idx::Vararg{Int,2})
    i,j = idx
    return vov.vecs[j][i]
end

function setindex!(vov::VOV, v, idx::Vararg{Int,2})
    i,j = idx
    vov.vecs[j][i] = v
end

function eachcol(vov::VOV)
    return vov.vecs
end

"""
`addRow!(vov::VectorOfVectors)`

Add a row to the vector of vectors by appending
`zero(T)` to each vector.
"""
function addRow!(vov::VOV{T}) where T
    for vec in vov.vecs 
        push!(vec, zero(T))
    end
    vov.size[1] += 1
end

"""
`addRow!(vov::VectorOfVectors, row::AbstractVector)`

Add the vector `row` to the last column of `vov`.
"""
function addRow!(vov::VOV{T}, row::AbstractVector{T}) where T
    if length(row) != vov.size[2]
        error("New row must have the correct length")
    end
    for (i,vec) in enumerate(vov.vecs) 
        push!(vec, row[i])
    end
    vov.size[1] += 1
    nothing
end

"""
`removeRow!(vov::VectorOfVectors)`

Remove a row from the vector of vectors.
"""
function removeRow!(vov::VOV{T}) where T
    for vec in vov.vecs 
        pop!(vec)
    end
    vov.size[1] -= 1
    nothing
end

"""
`addCol!(vov::VectorOfVectors)`

Add a column to the vector of vectors by appending
`zeros(T,nrow)` to `vov.vecs`.
"""
function addCol!(vov::VOV{T}) where T
    nrows = vov.size[1]
    push!(vov.vecs, zeros(T, nrows))
    vov.size[2] += 1
    nothing
end

"""
`addCol!(vov::VectorOfVectors, col::AbstractVector)`

Add the vector `row` to the last column of `vov`.
"""
function addCol!(vov::VOV{T}, col::AbstractVector) where T
    nrows = vov.size[1]
    if length(col) != nrows
        error("New column must have the correct length")
    end
    push!(vov.vecs, col)
    vov.size[2] += 1
    nothing
end

"""
`removeCol!(vov::VectorOfVectors)`

Remove a column from the vector of vectors.
"""
function removeCol!(vov::VOV{T}) where T
    pop!(vov.vecs)
    vov.size[2] -= 1
    nothing
end