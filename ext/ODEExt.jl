module ODEExt

using ModelOrderReductionToolkit
using OrdinaryDiffEq: ODEProblem

function f_lti(dx, x, (model,u), t)
    mul!(dx, model.A, x)
    mul!(dx, model.B, u(t), 1, 1)
    ldiv!(model.E, dx)
end

"""
`to_ode_problem(model[, p=nothing; u=(t->zeros(size(model.B, 2))), x0=0.0, tspan=(0,1)])`

Creates an `ODEProblem` for the `model <: LTISystem` for a given input `u(t)`.
Note that this is the ODE for the state variable `x`. Once have formed the solution
object, will have to multiply by `model.C` to get the output `y`. Note that
`DifferentialEquations.jl` names the output `u`, which for this problem is the state
variable `x`, not the input `u`.
"""
function ModelOrderReductionToolkit.to_ode_problem(model::LTIModel, p=nothing; u=(t->zeros(size(model.B, 2))), x0::Union{Number,AbstractVector}=0.0, tspan=(0,1))
    if !isnothing(p)
        model(p)
    end
    if isa(x0, Number)
        x0 = x0 .* ones(output_type(model), output_length(model))
    end
    ode_p = (model,u)
    if !isa(model.E, UniformScaling)
        model.E = factorize(model.E)
    end 
    return ODEProblem(f_lti, x0, tspan, ode_p)
end

end