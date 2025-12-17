"""
`StationaryModel{DIM}`

Abstract type for parametric, stationary models of the form
`L(u(p), p) = f(p)` where `p` is some parameter and `L` is some
(possibly nonlinear) operator, and `u(p)` is the parameter-dependent
solution. Since we are focused on discretized problems, `u(p)` and `f(p)`
are vectors, and `L` is an invertible operator between vector spaces.
"""
abstract type StationaryModel{Int} end

function (m::StationaryModel)(p, i::Int=1)
    error("Must implement solution model(p, i::Int=1) for StationaryModel")
end

function output_type(m::StationaryModel)
    error("Must implement output_type for StationaryModel")
end

function output_length(m::StationaryModel)
    error("Must implement output_length for StationaryModel")
end

"""
`NonstationaryModel`

Abstract type for parametric, nonstationary models of the form
`∂_t u(t,p) = L(u(t,p), p) + f(p)` where `p` is some parameter and `L` is some
(possibly nonlinear) operator, and `u(t,p)` is the time and parameter-dependent
solution. Since we are focused on discretized problems, `u(p)` and `f(p)`
are vectors, and `L` is an operator between vector spaces.
"""
abstract type NonstationaryModel end

function (m::NonstationaryModel)(p)
    error("Must implement solution model(p) for NonstationaryModel")
end

"""
`to_ode_problem(model::NonstationaryModel[, p=nothing; u=(t->0), x0=0.0, tspan=(0,1)])`

Creates an `ODEProblem` for the `model <: NonstationaryModel` for a given input `u(t)`.
To use this method requires importing `OrdinaryDiffEq` as this functionality lives 
in an extension. 
"""
function to_ode_problem(m::NonstationaryModel, p=nothing; u=(t->0), x0=0.0, tspan=(0,1))
    error("Must implement to_ode_problem for NonstationaryModel; must import OrdinaryDiffEq to use this method")
end

function output_type(m::NonstationaryModel)
    error("Must implement output_type for NonstationaryModel")
end

function output_length(m::NonstationaryModel)
    error("Must implement output_length for NonstationaryModel")
end