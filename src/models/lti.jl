"""
`model = LTIModel(A_in, B_in, C_in, D_in=0, E_in=I) <: NonstationaryModel`

`model = LTIModel(lti<:AbstractStateSpace) <: NonstationaryModel`

`model = LTIModel(lti<:AbstractDescriptorStateSpace) <: NonstationaryModel`

Struct for containing a parameterized LTI model
```math
E(p) x'(t,p) = A(p) x(t,p) + B(p) u(t)
y(t,p) = C(p) x(t,p) + D(p) u(t)
```
with affine parameter dependence where `u(t)` is some
input signal. Can initialize the system to a given 
parameter `p` by calling `model(p)`.
"""
mutable struct LTIModel <: NonstationaryModel
    A::AbstractMatrix
    B::AbstractMatrix
    C::AbstractMatrix
    D::AbstractMatrix
    E::Union{AbstractMatrix,UniformScaling}
    Ap::Union{APArray,Nothing}
    Bp::Union{APArray,Nothing}
    Cp::Union{APArray,Nothing}
    Dp::Union{APArray,Nothing}
    Ep::Union{APArray,Nothing}
end

function LTIModel(A_in::Union{AbstractMatrix,APArray}, 
                  B_in::Union{AbstractMatrix,APArray},  
                  C_in::Union{AbstractMatrix,APArray}, 
                  D_in::Union{AbstractMatrix,APArray,Real}=0,
                  E_in::Union{UniformScaling,AbstractMatrix,APArray}=I)
    arrs = Union{AbstractMatrix,UniformScaling}[]
    aparrs = Union{APArray,Nothing}[]
    for (i,mat_in) in enumerate((A_in, B_in, C_in, D_in, E_in))
        if mat_in isa APArray
            T = typeof(prod(zero.(eltype.(mat_in.arrays))))
            if all(issparse.(mat_in.arrays))
                push!(arrs, spzeros(T, size(mat_in.arrays[1])))
            else
                push!(arrs, Matrix{T}(undef, size(mat_in.arrays[1])))
            end
            push!(aparrs, mat_in)
        elseif (i == 4) && mat_in isa Real
            push!(arrs, mat_in .* ones(typeof(D_in), size(arrs[3], 1), size(arrs[2], 2)))
            push!(aparrs, nothing)
        elseif (i == 5) && mat_in isa UniformScaling
            push!(arrs, mat_in)
            push!(aparrs, nothing)
        else
            push!(arrs, mat_in)
            push!(aparrs, nothing)
        end
    end
    return LTIModel(arrs..., aparrs...)
end

import Base: -

function -(m1::LTIModel, m2::LTIModel)
    n1_out = size(m1.C, 1)
    n2_out = size(m2.C, 1)
    @assert n1_out == n2_out
    n1 = size(m1.A, 1)
    n2 = size(m2.A, 1)
    n1_in = size(m1.B, 2)
    n2_in = size(m2.B, 2)
    n_in = max(n1_in, n2_in)
    # A is block diagonal of A's
    A_in = begin
        if isnothing(m1.Ap) && isnothing(m2.Ap)
            vcat(hcat(m1.A, spzeros(n1, n2)), hcat(spzeros(n2, n1), m2.A))
        elseif isnothing(m2.Ap)
            arrs = [vcat(hcat(A, spzeros(n1, n2)), hcat(spzeros(n2, n1), spzeros(n2, n2))) for A in m1.Ap.arrays]
            push!(arrs, vcat(hcat(spzeros(n1, n1), spzeros(n1, n2)), hcat(spzeros(n2, n1), m2.A)))
            makeθi = if m1.Ap.precompθ
                p -> [m1.Ap(p) ; [1]]
            else
                (p,i) -> i <= length(m1.Ap.arrays) ? m1.Ap.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        elseif isnothing(m1.Ap)
            arrs = [vcat(hcat(spzeros(n1, n1), spzeros(n1, n2)), hcat(spzeros(n2, n1), A)) for A in m2.Ap.arrays]
            push!(arrs, vcat(hcat(m1.A, spzeros(n1, n2)), hcat(spzeros(n2, n1), spzeros(n2, n2))))
            makeθi = if m2.Ap.precompθ
                p -> [m2.Ap(p) ; [1]]
            else
                (p,i) -> i <= length(m2.Ap.arrays) ? m2.Ap.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        else
            arrs = [vcat(hcat(A, spzeros(n1, n2)), hcat(spzeros(n2, n1), spzeros(n2, n2))) for A in m1.Ap.arrays]
            append!(arrs, [vcat(hcat(spzeros(n1, n1), spzeros(n1, n2)), hcat(spzeros(n2, n1), A)) for A in m2.Ap.arrays])
            makeθi = if m1.Ap.precompθ && m2.Ap.precompθ
                p -> [m1.Ap(p) ; m2.Ap(p)]
            elseif m2.Ap.precompθ
                p -> [[m1.Ap(p,i) for i in eachindex(m1.Ap.arrays)] ; m2.Ap(p)]
            elseif m1.Ap.precompθ
                p -> [m1.Ap(p) ; [m2.Ap(p,i) for i in eachindex(m2.Ap.arrays)]]
            else
                (p,i) -> i <= length(m1.Ap.arrays) ? m1.Ap.makeθi(p,i) : m2.Ap.makeθi(p,i - length(m1.Ap.arrays))
            end
            APArray(arrs, makeθi)
        end
    end
    # E is block diagonal of E's
    E_in = begin
        if isnothing(m1.Ep) && isnothing(m2.Ep)
            if isa(m1.E, UniformScaling) && isa(m2.E, UniformScaling) && m1.E == m2.E
                m1.E
            else
                vcat(hcat(m1.E, spzeros(n1, n2)), hcat(spzeros(n2, n1), m2.E))
            end
        elseif isnothing(m2.Ep)
            arrs = [vcat(hcat(E, spzeros(n1, n2)), hcat(spzeros(n2, n1), spzeros(n2, n2))) for E in m1.Ep.arrays]
            push!(arrs, vcat(hcat(spzeros(n1, n1), spzeros(n1, n2)), hcat(spzeros(n2, n1), m2.E)))
            makeθi = if m1.Ep.precompθ
                p -> [m1.Ep(p) ; [1]]
            else
                (p,i) -> i <= length(m1.Ep.arrays) ? m1.Ep.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        elseif isnothing(m1.Ep)
            arrs = [vcat(hcat(spzeros(n1, n1), spzeros(n1, n2)), hcat(spzeros(n2, n1), E)) for E in m2.Ep.arrays]
            push!(arrs, vcat(hcat(m1.E, spzeros(n1, n2)), hcat(spzeros(n2, n1), spzeros(n2, n2))))
            makeθi = if m2.Ep.precompθ
                p -> [m2.Ep(p) ; [1]]
            else
                (p,i) -> i <= length(m2.Ep.arrays) ? m2.Ep.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        else
            arrs = [vcat(hcat(E, spzeros(n1, n2)), hcat(spzeros(n2, n1), spzeros(n2, n2))) for E in m1.Ep.arrays]
            append!(arrs, [vcat(hcat(spzeros(n1, n1), spzeros(n1, n2)), hcat(spzeros(n2, n1), A)) for A in m2.Ep.arrays])
            makeθi = if m1.Ep.precompθ && m2.Ep.precompθ
                p -> [m1.Ep(p) ; m2.Ep(p)]
            elseif m2.Ep.precompθ
                p -> [[m1.Ep(p,i) for i in eachindex(m1.Ep.arrays)] ; m2.Ep(p)]
            elseif m1.Ep.precompθ
                p -> [m1.Ep(p) ; [m2.Ep(p,i) for i in eachindex(m2.Ep.arrays)]]
            else
                (p,i) -> i <= length(m1.Ep.arrays) ? m1.Ep.makeθi(p,i) : m2.Ep.makeθi(p,i - length(m1.Ep.arrays))
            end
            APArray(arrs, makeθi)
        end
    end
    # B is vstacked B's
    B_in = begin
        if isnothing(m1.Bp) && isnothing(m2.Bp)
            vcat(hcat(m1.B, spzeros(n1, n_in - n1_in)), hcat(m2.B, spzeros(n2, n_in - n2_in)))
        elseif isnothing(m2.Bp)
            arrs = [vcat(hcat(B, spzeros(n1, n_in - n1_in)), hcat(spzeros(n2, n2_in), spzeros(n2, n_in - n2_in))) for B in m1.Bp.arrays]
            push!(arrs, vcat(hcat(spzeros(n1, n1_in), spzeros(n1, n_in - n1_in)), hcat(m2.B, spzeros(n2, n_in - n2_in))))
            makeθi = if m1.Bp.precompθ
                p -> [m1.Bp(p) ; [1]]
            else
                (p,i) -> i <= length(m1.Bp.arrays) ? m1.Bp.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        elseif isnothing(m1.Bp)
            arrs = [vcat(hcat(spzeros(n1, n1_in), spzeros(n1, n_in - n1_in)), hcat(B, spzeros(n2, n_in - n2_in))) for B in m2.Bp.arrays]
            push!(arrs, vcat(hcat(m1.B, spzeros(n1, n_in - n1_in)), hcat(spzeros(n2, n2_in), spzeros(n2, n_in - n2_in))))
            makeθi = if m2.Bp.precompθ
                p -> [m2.Bp(p) ; [1]]
            else
                (p,i) -> i <= length(m2.Bp.arrays) ? m2.Bp.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        else
            arrs = [vcat(hcat(B, spzeros(n1, n_in - n1_in)), hcat(spzeros(n2, n2_in), spzeros(n2, n_in - n2_in))) for B in m1.Bp.arrays]
            append!(arrs, [vcat(hcat(spzeros(n1, n1_in), spzeros(n1, n_in - n1_in)), hcat(B, spzeros(n2, n_in - n2_in))) for B in m2.Bp.arrays])
            makeθi = if m1.Bp.precompθ && m2.Bp.precompθ
                p -> [m1.Bp(p) ; m2.Bp(p)]
            elseif m2.Bp.precompθ
                p -> [[m1.Bp(p,i) for i in eachindex(m1.Bp.arrays)] ; m2.Bp(p)]
            elseif m1.Bp.precompθ
                p -> [m1.Bp(p) ; [m2.Bp(p,i) for i in eachindex(m2.Bp.arrays)]]
            else
                (p,i) -> i <= length(m1.Bp.arrays) ? m1.Bp.makeθi(p,i) : m2.Bp.makeθi(p,i - length(m1.Bp.arrays))
            end
            APArray(arrs, makeθi)
        end
    end
    # C is hstacked C's
    C_in = begin
        if isnothing(m1.Cp) && isnothing(m2.Cp)
            hcat(m1.C, -1 .* m2.C)
        elseif isnothing(m2.Cp)
            arrs = [hcat(C, spzeros(size(m2.C))) for C in m1.Cp.arrays]
            push!(arrs, hcat(spzeros(size(m1.C)), m2.C))
            makeθi = if m1.Cp.precompθ
                p -> [m1.Cp(p) ; [-1]]
            else
                (p,i) -> i <= length(m1.Cp.arrays) ? m1.Cp.makeθi(p,i) : -1
            end
            APArray(arrs, makeθi)
        elseif isnothing(m1.Cp)
            arrs = [hcat(spzeros(size(m1.C), C)) for C in m2.Cp.arrays]
            push!(arrs, hcat(spzeros(size(m1.C)), m2.C))
            makeθi = if m2.Cp.precompθ
                p -> [-1 .* m2.Cp(p) ; [1]]
            else
                (p,i) -> i <= length(m2.Cp.arrays) ? -1 * m2.Cp.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        else
            arrs = [hcat(C, spzeros(size(m2.C))) for C in m1.Cp.arrays]
            append!(arrs, [hcat(spzeros(size(m1.C), C)) for C in m2.Cp.arrays])
            makeθi = if m1.Cp.precompθ && m2.Cp.precompθ
                p -> [m1.Cp(p) ; -1 .* m2.Cp(p)]
            elseif m2.Cp.precompθ
                p -> [[m1.Cp(p,i) for i in eachindex(m1.Cp.arrays)] ; -1 .* m2.Cp(p)]
            elseif m1.Cp.precompθ
                p -> [m1.Cp(p) ; [-1 * m2.Cp(p,i) for i in eachindex(m2.Cp.arrays)]]
            else
                (p,i) -> i <= length(m1.Cp.arrays) ? m1.Cp.makeθi(p,i) : -1 * m2.Cp.makeθi(p,i - length(m1.Cp.arrays))
            end
            APArray(arrs, makeθi)
        end
    end
    # D is difference between Ds
    D_in = begin
        if isnothing(m1.Dp) && isnothing(m2.Dp)
            hcat(m1.D, zeros(n1_out, n_in - n1_in)) .- hcat(m2.D, zeros(n1_out, n_in - n2_in)) 
        elseif isnothing(m2.Dp)
            arrs = [hcat(D, zeros(n1_out, n_in - n1_in)) for D in m1.Dp.arrays]
            push!(arrs, hcat(m2.D, zeros(n1_out, n_in - n2_in)))
            makeθi = if m1.Dp.precompθ
                p -> [m1.Dp(p) ; [-1]]
            else
                (p,i) -> i <= length(m1.Dp.arrays) ? m1.Dp.makeθi(p,i) : -1
            end
            APArray(arrs, makeθi)
        elseif isnothing(m1.Dp)
            arrs = [hcat(D, zeros(n1_out, n_in - n2_in)) for D in m2.Dp.arrays]
            push!(arrs, hcat(m1.D, zeros(n1_out, n_in - n1_in)))
            makeθi = if m2.Dp.precompθ
                p -> [-1 .* m2.Dp(p) ; [1]]
            else
                (p,i) -> i <= length(m2.Dp.arrays) ? -1 * m2.Dp.makeθi(p,i) : 1
            end
            APArray(arrs, makeθi)
        else
            arrs = [hcat(D, zeros(n1_out, n_in - n1_in)) for D in m1.Dp.arrays]
            append!(arrs, [hcat(D, zeros(n1_out, n_in - n2_in)) for D in m2.Dp.arrays])
            makeθi = if m1.Dp.precompθ && m2.Dp.precompθ
                p -> [m1.Dp(p) ; -1 .* m2.Dp(p)]
            elseif m2.Dp.precompθ
                p -> [[m1.Dp(p,i) for i in eachindex(m1.Dp.arrays)] ; -1 .* m2.Dp(p)]
            elseif m1.Dp.precompθ
                p -> [m1.Dp(p) ; [-1 * m2.Dp(p,i) for i in eachindex(m2.Dp.arrays)]]
            else
                (p,i) -> i <= length(m1.Dp.arrays) ? m1.Dp.makeθi(p,i) : -1 * m2.Dp.makeθi(p,i - length(m1.Dp.arrays))
            end
            APArray(arrs, makeθi)
        end
    end
    return LTIModel(A_in, B_in, C_in, D_in, E_in)
end

function Base.show(io::Core.IO, model::LTIModel)
    res  = "LTI Model with state length $(size(model.A, 1)), input length $(size(model.B, 2)), and output length $(size(model.C, 1))"
    println(io, res)
    for (str, matp, mat) in [("A", model.Ap, model.A), ("B", model.Bp, model.B), ("C", model.Cp, model.C), ("D", model.Dp, model.D)]
        print(io, "$str - ")
        if isnothing(matp)
            println(io, "$(size(mat)) $(typeof(mat))")
        else
            println(io, matp)
        end
    end
    print(io, "E - ")
    if isnothing(model.Ep)
        if isa(model.E, UniformScaling)
            print(io, model.E)
        else
            print(io, "$(size(model.E)) $(typeof(model.E))")
        end
    else
        print(io, Ep)
    end
end

function (model::LTIModel)(p)
    if !isnothing(model.Ap)
        formArray!(model.Ap,model.A,p)
    end
    if !isnothing(model.Bp)
        formArray!(model.Bp,model.B,p)
    end
    if !isnothing(model.Cp)
        formArray!(model.Cp,model.C,p)
    end
    if !isnothing(model.Dp)
        formArray!(model.Dp,model.D,p)
    end
    if !isnothing(model.Ep)
        formArray!(model.Ep,model.E,p)
    end
end

function output_length(model::LTIModel)
    return size(model.A,2)
end

function output_type(model::LTIModel)
    T1 = eltype(model.A)
    T2 = eltype(model.B)
    T3 = eltype(model.E)
    return typeof(zero(T1) * zero(T2) * zero(T3))
end

function is_parameterized(model::LTIModel)
    return !(isnothing(model.Ap) &&
             isnothing(model.Bp) && 
             isnothing(model.Cp) &&
             isnothing(model.Dp) &&
             isnothing(model.Ep))
end

"""
`bode(model::LTIModel, ω::Real[, p=nothing; first=true])`

Returns the transfer function evaluated at `s=im*ω`,
`C * (sE - A)^(-1) B + D`, evaluated at the `[1,1]` entry if `first==true`.
"""
function bode(model::LTIModel, ω::Real, p=nothing; first=true)
    if !isnothing(p)
        model(p)
    end
    s = ω*im
    res = begin
        if issparse(model.B)
            if issparse(model.C)
                if size(model.B, 2) <= size(model.C, 1)
                    model.C * ((s*model.E - model.A) \ collect(model.B)) .+ model.D
                else
                    (collect(model.C) / (s*model.E - model.A)) * model.B .+ model.D
                end
            else
                (model.C / (s*model.E - model.A)) * model.B .+ model.D
            end
        elseif issparse(model.C)
            model.C * ((s*model.E - model.A) \ model.B) .+ model.D
        else
            if size(model.B, 2) <= size(model.C, 1)
                model.C * ((s*model.E - model.A) \ model.B) .+ model.D
            else
                (model.C / (s*model.E - model.A)) * model.B .+ model.D
            end
        end
    end
    if first
        return res[1,1]
    else
        return res
    end
end

"""
`bode(model::LTIModel, ωs::AbstractVector{<:Union{AbstractVector,Real}}[, p=nothing; first=true])`

Returns the transfer function evaluated at `s=im*ω`
for `ω in ωs`, evaluated at the `[1,1]` entry if `first==true`.
"""
function bode(model::LTIModel, ωs::AbstractVector{<:Union{AbstractVector,Real}}, p=nothing; first=true)
    if !isnothing(p)
        model(p)
    end
    return [bode(model, ω[1], first=first) for ω in ωs]
end

"""
`poles_and_vectors(model::LTIModel[, p=nothing])`

Returns the eigenvalues and eigenvectors `Ax=λEx` of the LTI system 
`model` at the parameter `p`. Warning: relies on dense LA, do not use
for huge sparse systems.
"""
function poles_and_vectors(model::LTIModel, p=nothing)
    if !isnothing(p)
        model(p)
    end
    A = model.A
    E = model.E
    if isa(E, UniformScaling)
        return eigen(Matrix(E \ A))
    else
        return eigen(Matrix(A), Matrix(E))
    end
end

"""
`poles(model::LTIModel[, p=nothing])`

Returns the poles (eigenvalues `Ax=λEx`) of the LTI system `model`
at the parameter `p`. Warning: relies on dense LA, do not use
for huge sparse systems.
"""
function poles(model::LTIModel, p=nothing)
    return poles_and_vectors(model, p).values
end

"""
`to_ss(model[, p=nothing])`

Must import `ControlSystems.jl` for this functionality as
its definition lives in an extension.

Initializes the model to the parameter `p` if passed
in, then returns a `ControlSystems.jl` `StateSpace`
object. 
"""
function to_ss end

"""
`to_dss(model[, p=nothing])`

Must import `DescriptorSystems.jl` for this functionality as
its definition lives in an extension.

Initializes the model to the parameter `p` if passed
in, then returns a `DescriptorSystems.jl`
`DescriptorStateSpace` object. 
"""
function to_dss end

"""
`frequency_model = to_frequency_domain(model, logfreq=false)`

Assuming null initial conditions, uses the Laplace transform
to convert the `LTIModel` into a `LinearMatrixModel` for which the
first element of the parameter vector is the imaginary part of
the frequency variable. The model is of the form
`sE(p) X = A(p) X + B(p)` where `X` is the Laplace variable
for `X` and `s = 0 + iω`. If `logfreq=true`, then scales ω
such that `p[1] = log10(ω)`.
"""
function to_frequency_domain(model::LTIModel, logfreq=false) # TODO: Add Re(s) != 0 argument?
    scale = logfreq ? exp10 : identity
    imEp = begin 
        if isnothing(model.Ep)
            arrs = if isa(model.E, UniformScaling)
                [spdiagm(im * model.E.λ .* ones(ComplexF64, size(model.A, 1)))]
            else
                [im .* model.E]
            end
            APArray(arrs, i -> scale(p[1]))
        else
            APArray(im .* model.Ep.arrays, model.Ep.makeθi)
        end
    end

    sEmAp = begin
        if isnothing(model.Ap)
            sImAs = [imEp.arrays..., (1 + 0im) .* model.A]
            makeθImAis = (p,i) -> (i <= length(imEp.arrays)) ? scale(p[1]) : -1.0
            APArray(sImAs, makeθImAis)
        else
            # Paramater dependence in A
            sImAs = [imEp.arrays..., ((1 + 0im) .* model.Ap.arrays)...]
            makeθImAis = (p,i) -> (i <= length(imEp.arrays)) ? scale(p[1]) : (-1.0 .* model.Ap.makeθi(view(p,2:length(p)), i-length(imEp.arrays)))
            makeθImAis = (p,i) -> (i == 1) ? scale(p[1]) : (-1.0 .* model.Ap.makeθi(view(p,2:length(p)), i-length(imEp.arrays)))
            APArray(sImAs, makeθImAis)
        end
    end

    B_in = isnothing(model.Bp) ? model.B : model.Bp
    
    return LinearMatrixModel(sEmAp, B_in)
end

"""
`galerkin_project(model, V[, W=V; WTEVisI=false, r=-1])`

Performs Galerkin projection on the `model <: LTIModel` and
returns a new `LTIModel`. If `W` and `V` are semiunitary, 
aka `W'V=I`
`WTEVisI==true`, then assumes that .
"""
function galerkin_project(model::LTIModel, V::AbstractMatrix, W::AbstractMatrix=V; WTEVisI=false, r=-1)
    N, n = size(V)
    if 0 < r && r < min(N, n)
        V = view(V, 1:N, 1:r)
        W = view(W, 1:N, 1:r)
    end

    A_in = begin
        if isnothing(model.Ap)
            VectorOfVectors(W' * model.A * V)
        else
            reduced_Ais = [VectorOfVectors(W' * Ai * V) for Ai in model.Ap.arrays]
            APArray(reduced_Ais, model.Ap.makeθi)
        end
    end

    B_in = begin
        if isnothing(model.Bp)
            VectorOfVectors(W' * model.B)
        else
            reduced_Bis = [VectorOfVectors(W' * Bi) for Bi in model.Bp.arrays]
            APArray(reduced_Bis, model.Bp.makeθi)
        end
    end

    C_in = begin
        if isnothing(model.Cp)
            VectorOfVectors(model.C * V)
        else
            reduced_Cis = [VectorOfVectors(Ci * V) for Ci in model.Cp.arrays]
            APArray(reduced_Cis, model.Cp.makeθi)
        end
    end

    D_in = begin
        if isnothing(model.Dp)
            model.D
        else
            model.Dp
        end
    end
    
    E_in = begin
        if isnothing(model.Ep)
            if WTEVisI
                I
            else
                VectorOfVectors(W' * model.E * V)
            end
        else
            reduced_Eis = [VectorOfVectors(W' * Ei * V) for Ei in model.Ep.arrays]
            APArray(reduced_Eis, model.Ep.makeθi)
        end
    end

    return LTIModel(A_in, B_in, C_in, D_in, E_in)
end