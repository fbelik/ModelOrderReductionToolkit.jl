"""
`model = LTIModel(A_in, B_in, C_in, D_in=0, E_in=I) <: NonstationaryModel`
`model = LTIModel(lti<:AbstractStateSpace) <: NonstationaryModel`
`model = LTIModel(lti<:AbstractDescriptorStateSpace) <: NonstationaryModel`

Struct for containing a parameterized LTI model
`E(p) x'(t,p) = A(p) x(t,p) + B(p) u(t)`
`y(t,p) = C(p) x(t,p) + D(p) u(t)`
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
    (A, Ap) = begin 
        if A_in isa APArray
            T = typeof(prod(zero.(eltype.(A_in.arrays))))
            (Matrix{T}(undef, size(A_in.arrays[1])), A_in)
        else
            (A_in, nothing)
        end
    end
    (B, Bp) = begin 
        if B_in isa APArray
            T = typeof(prod(zero.(eltype.(B_in.arrays))))
            (Matrix{T}(undef, size(B_in.arrays[1])), B_in)
        else
            (B_in, nothing)
        end
    end
    (C, Cp) = begin 
        if C_in isa APArray
            T = typeof(prod(zero.(eltype.(C_in.arrays))))
            (Matrix{T}(undef, size(C_in.arrays[1])), C_in)
        else
            (C_in, nothing)
        end
    end
    (D, Dp) = begin 
        if D_in isa APArray
            T = typeof(prod(zero.(eltype.(D_in.arrays))))
            (Matrix{T}(undef, size(D_in.arrays[1])), D_in)
        elseif D_in isa Real
            (D_in .* ones(typeof(D_in), size(C, 1), size(B, 2)), nothing)
        else
            (D_in, nothing)
        end
    end
    (E, Ep) = begin 
        if E_in isa APArray
            T = typeof(prod(zero.(eltype.(E_in.arrays))))
            (Matrix{T}(undef, size(E_in.arrays[1])), E_in)
        else
            (E_in, nothing)
        end
    end
    return LTIModel(A, B, C, D, E, Ap, Bp, Cp, Dp, Ep)
end

function LTIModel(lti::AbstractStateSpace)
    return LTIModel(lti.A, lti.B, lti.C, lti.D)
end

function LTIModel(lti::AbstractDescriptorStateSpace)
    return LTIModel(lti.A, lti.B, lti.C, lti.D, lti.E)
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

function f_lti(dx, x, p, t)
    A, B, E, u = p
    # dx .= E \ (A * x .+ B * u(t))
    mul!(dx, A, x)
    mul!(dx, B, u(t), 1, 1)
    ldiv!(E, dx)
end

"""
`to_ode_problem(model[, p=nothing; u=(t->zeros(size(model.B, 2))), x0=0.0, tspan=(0,1)])`

Creates an `ODEProblem` for the `model <: LTISystem` for a given input `u(t)`.
Note that this is the ODE for the state variable `x`. Once have formed the solution
object, will have to multiply by `model.C` to get the output `y`. Note that
`DifferentialEquations.jl` names the output `u`, which for this problem is the state
variable `x`, not the input `u`.
"""
function to_ode_problem(model::LTIModel, p=nothing; u=(t->zeros(size(model.B, 2))), x0::Union{Number,AbstractVector}=0.0, tspan=(0,1))
    if !isnothing(p)
        model(p)
    end
    if isa(x0, Number)
        x0 = x0 .* ones(output_type(model), output_length(model))
    end
    E = isa(model.E, VectorOfVectors) ? Matrix(model.E) : model.E
    ode_p = (Matrix(model.A), Matrix(model.B), E, u)
    return ODEProblem(f_lti, x0, tspan, ode_p)
end

"""
`to_ss(model[, p=nothing])`

Initializes the model to the parameter `p` if passed
in, then returns a `ControlSystems.jl` `StateSpace`
object. 
"""
function to_ss(model::LTIModel, p=nothing)
    if !isnothing(p)
        model(p)
    end
    return ss(model.A, model.B, model.C, model.D)
end

"""
`to_dss(model[, p=nothing])`

Initializes the model to the parameter `p` if passed
in, then returns a `DescriptorSystems.jl`
`DescriptorStateSpace` object. 
"""
function to_dss(model::LTIModel, p=nothing)
    if !isnothing(p)
        model(p)
    end
    return dss(Matrix(model.A), isa(model.E, UniformScaling) ? model.E : Matrix(model.E), Matrix(model.B), Matrix(model.C), Matrix(model.D))
end

"""
`frequency_model = to_frequency_domain(model)`

Assuming null initial conditions, uses the Laplace transform
to convert the `LTIModel` into a `LinearMatrixModel` for which the
first element of the parameter vector is the imaginary part of
the frequency variable. The model is of the form
`sE(p) X = A(p) X + B(p)` where `X` is the Laplace variable
for `X` and `s = 0 + iω`
"""
function to_frequency_domain(model::LTIModel) # TODO: Add Re(s) != 0 argument?
    imEp_arrs = begin 
        if isnothing(model.Ep)
            if isa(model.E, UniformScaling)
                [spdiagm(im * model.E.λ .* ones(ComplexF64, size(model.A, 1)))]
            else
                [im .* model.E]
            end
        else
            APArray(im .* model.Ep.arrays, model.Ep.makeθi)
        end
    end

    sEmAp = begin
        if isnothing(model.Ap)
            sImAs = [imEp_arrs..., (1 + 0im) .* model.A]
            makeθImAis = (p,i) -> (i <= length(imEp_arrs)) ? p[1] : -1.0
            APArray(sImAs, makeθImAis)
        else
            # Paramater dependence in A
            sImAs = [imEp_arrs..., ((1 + 0im) .* model.Ap.arrays)...]
            makeθImAis = (p,i) -> (i <= length(imEp_arrs)) ? p[1] : (-1.0 .* model.Ap.makeθi(view(p,2:length(p)), i-length(imEp_arrs)))
            APArray(sImAs, makeθImAis)
        end
    end

    B_in = isnothing(model.Bp) ? model.B : model.Bp
    
    return LinearMatrixModel(sEmAp, B_in)
end

"""
`galerkin_project(model, V[, W=V, ])`
"""
function galerkin_project(model::LTIModel, V::AbstractMatrix, W::AbstractMatrix=V; r=-1)
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
            if isa(model.E, UniformScaling)
                model.E
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