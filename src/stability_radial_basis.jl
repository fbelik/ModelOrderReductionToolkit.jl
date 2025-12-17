gaussian_rbf(r,ϵ=1) = exp(-(ϵ*r)^2)
multiquadric_rbf(r,ϵ=1) = sqrt(1+(ϵ*r)^2)
inverse_quadratic_rbf(r,ϵ=1) = (1+(ϵ*r)^2)^(-1)
inverse_multiquadric_rbf(r,ϵ=1) = sqrt(1+(ϵ*r)^2)^(-1)
thin_plate_spline_rbf(r) = r^2*log(r)
bump_rbf(r,ϵ=1) = r >= (1/ϵ) ? 0 : exp(-1/(1-(ϵ*r)^2))

"""
`Sigma_Min_RBF`

A mutable struct used for approximating
the minimum singular value of a matrix 
with parametric dependence `A(p)`. Given 
a new parameter value `p`, approximate
`σ_min(A(p))` with `sigma_min_rbf(p)`.
"""
mutable struct Sigma_Min_RBF <: Function
    NP::Int
    P::Int
    b::Vector
    coef::Vector
    ϕ::Function
    params::AbstractVector
end

function (sigma_min_rbf::Sigma_Min_RBF)(p)
    logσ = 0
    for i in 1:sigma_min_rbf.NP
        logσ += sigma_min_rbf.coef[i] * sigma_min_rbf.ϕ(norm(p .- sigma_min_rbf.params[i]))
    end
    for i in 1:sigma_min_rbf.P
        logσ += sigma_min_rbf.coef[i+sigma_min_rbf.NP] * p[i]
    end
    logσ += sigma_min_rbf.coef[end]
    return exp(logσ)
end

"""
`min_sigma_rbf(params, Ais, makeθAi[, ϕ=gaussian_rbf])`

Method to form a interpolatory radial-basis function
functor to approximate the minimum singular value of a
parametrized matrix `A(p)`. Pass as input either a matrix
`params` where each column is a parameter vector, or a vector 
of vectors `params` where each vector is a parameter. 

Optional argument for the radial-basis function `ϕ`, which
defaults to the Gaussian `ϕ(r) = exp(-r^2)`.

Returns a functor `sigma_min_rbf <: Sigma_Min_RBF` such that given a new
parameter vector `p`, `sigma_min_rbf(p)` returns an approximation
to the minimum singular value of `A(p)`.

Offline, solves for the minimum singular value for each parameter
in `params`, and uses this to form an interpolatory approximation
in the form of
`log(σ_min(A(p))) ≈ ω_0 + ∑ (ω_i p_i) + ∑ (γ_i ϕ(p - p_i))`
where `p_i` is the i'th parameter in `params`, and `ω_0`, `ω_i`, and
`γ_i` are determined by the given `params` such that the approximation
holds true for all `p_i`, and that `∑ γ_i = 0`, and that `∑ γ_i p_i[j] = 0`
for each `j`.
"""
function min_sigma_rbf(params::Union{Matrix,Vector},
                       Ap::Union{Function,APArray},
                       ϕ::Function=gaussian_rbf;
                       kmaxiter=1000,
                       noise=0,
                       progress=false)

    if typeof(params) <: Matrix
        P,NP = size(params)
        params = SVector{NP}([SVector{P}(params[:,i]) for i in 1:NP])
    else
        P = length(params[1])
        NP = length(params)
        params = SVector{NP}([SVector{P}(p) for p in params])
    end
    # Explicitly compute stability factors
    σ_mins = zeros(NP)
    if noise >= 1
        println("Beginning to explicitly compute stability factors")
        println("-----")
    end
    lP = length(params)
    lPten = ceil(Int, lP / 10)
    for i in (progress ? ProgressBar(eachindex(params)) : eachindex(params))
        p = params[i]
        # Compute minimum singular value
        A = Ap(p)
        σ_mins[i] = smallest_sval(A, kmaxiter=kmaxiter, noise=noise)
    end
    if noise >= 1
        println("-----")
        println("Completed explicitly computing stability factors")
    end
    # Form matrix M, NP+P+1×NP+P+1
    M = zeros(NP+P+1, NP+P+1)
    for i in 1:NP
        for j in 1:NP
            M[i,j] = ϕ(norm(params[i] .- params[j]))
        end
        for j in 1:P
            M[i,j+NP] = params[i][j]
        end
        M[i,NP+P+1] = 1
    end
    for i in 1:P
        for j in 1:NP
            M[i+NP,j] = params[j][i]
        end
    end
    for j in 1:NP
            M[NP+P+1,j] = 1
    end
    b = zeros(NP+P+1)
    for i in 1:NP
        b[i] = log(σ_mins[i])
    end
    coef = M \ b
    # Create and return functor
    sigma_min_rbf = Sigma_Min_RBF(NP, P, b, coef, ϕ, params)
    return sigma_min_rbf
end

"""
`update_sigma_rbf!(sigma_min_rbf, ϕ)`

Given a `Sigma_Min_RBF` object `sigma_min_rbf`, update its
RBF, `ϕ`, and update the coefficients.
"""
function update_sigma_rbf!(sigma_min_rbf::Sigma_Min_RBF,
                           ϕ::Function=gaussian_rbf)

    P = sigma_min_rbf.P
    NP = sigma_min_rbf.NP
    params = sigma_min_rbf.params
    # Form matrix M, NP+P+1×NP+P+1
    M = zeros(NP+P+1, NP+P+1)
    for i in 1:NP
        for j in 1:NP
            M[i,j] = ϕ(norm(params[i] .- params[j]))
        end
        for j in 1:P
            M[i,j+NP] = params[i][j]
        end
        M[i,NP+P+1] = 1
    end
    for i in 1:P
        for j in 1:NP
            M[i+NP,j] = params[j][i]
        end
    end
    for j in 1:NP
            M[NP+P+1,j] = 1
    end
    b = sigma_min_rbf.b
    coef = M \ b
    # Update functor
    sigma_min_rbf.ϕ = ϕ
    sigma_min_rbf.coef = coef
end

