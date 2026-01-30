"""
`reductor = BTReductor(model::LTIModel[, p=nothing; noise=0, iterative=nothing, maxdim=-1, lradi_eps=1e-6, dense_row_tol=1e-8])`

Balanced truncation reductor object for reducing an `LTIModel`. If parameter
`p` passed in, model first initialized to parameter value. `noise` determines
amount of printed output. If `isnothing(iterative)`, checks the size and sparsity
of the system whether or not to use an iterative method. If `iterative==true`, 
uses an iterative low-rank ADI method to solve for the Gramians (see Kurschner and Benner 2016 Dissertation Alg 4.3),
otherwise, uses MatrixEquations.jl to solve them densely. `maxdim` determines the 
maximum rank for the iterative solver, and `lradi_eps` determines a tolerance for
when the algorithm terminantes. If `iterative==false`, `dense_row_tol` is used to 
truncate the dense Gramians for quicker solving of the RB and HSVs.

The reachability Gramian can be computed by `reductor.R * reductor.R'`, and the 
observability Gramian by `reductor.L * reductor.L'`.

HSVs stored in `reductor.hs`.

Petrov-Galerkin test and trial spaces stored in `reductor.V` and `reductor.W` respectively.

Initialized-to parameter value stored in `reductor.p`.
"""
struct BTReductor
    model::LTIModel
    R::AbstractMatrix
    L::AbstractMatrix
    hs::AbstractVector
    V::AbstractMatrix
    W::AbstractMatrix
    p
end

function Base.show(io::Core.IO, reductor::BTReductor)
    res  = "BT Reductor with ability to reduce up to dimension $(min(size(reductor.V, 2), size(reductor.W, 2)))"
    println(io, res)
    print(io, "FOM: ")
    print(io, reductor.model)
    if !isnothing(reductor.p)
        println(io, "")
        print(io, "Initialized to parameter value $(reductor.p)")
    end
end

function BTReductor(model::LTIModel, p=nothing; noise=0, iterative::Union{Bool,Nothing}=nothing, maxdim=-1, lradi_eps=1e-6, dense_row_tol=1e-8)
    if !isnothing(p)
        model(p)
    elseif is_parameterized(model) && noise >= 1
        println("No parameter inputted, make sure the LTI model is not parameterized or already initialized to a parameter")
    end
    if isnothing(iterative)
        sparsity = issparse(model.A) && (isa(model.E, UniformScaling) || issparse(model.E))
        large = size(model.A,2) > 150
        iterative = sparsity && large
        if noise >= 1
            println("After looking at sparsity ($sparsity) and if sufficiently large (>150) system ($large), setting iterative=$iterative")
        end
    end
    R, L = begin
        if iterative
            (glyap_lradi_r(model.A, model.E, model.B, noise=noise, eps=lradi_eps, maxdim=maxdim),
             glyap_lradi_r(model.A', model.E, model.C', noise=noise, eps=lradi_eps, maxdim=maxdim))
        else
            E = isa(model.E, UniformScaling) ? model.E : Matrix(model.E')'
            _R = plyapc(Matrix(model.A')', E, Matrix(model.B')')'
            imax = findfirst(x -> norm(x) < dense_row_tol, eachcol(_R))
            _R = isnothing(imax) ? _R : view(_R, :, 1:imax)
            _L = plyapc(Matrix(model.A)', E, Matrix(model.C)')'
            imax = findfirst(x -> norm(x) < dense_row_tol, eachcol(_L))
            _L = isnothing(imax) ? _L : view(_L, :, 1:imax)
            (_R, _L)
        end
    end
    U, Δ, Z = svd(L' * model.E * R)
    T = R * Z * Diagonal(Δ .^ (-1/2))
    S = L * U * Diagonal(Δ .^ (-1/2))
    return BTReductor(model, R, L, Δ, T, S, p)
end

"""
`P = reachability_gramian(reductor::BTReductor)`

Helper method for forming the reachability Gramian.
"""
function reachability_gramian(reductor::BTReductor)
    return reductor.R * reductor.R'
end

"""
`P = reachability_gramian(model::LTIModel[, p=nothing; kwargs...])`

Helper method for forming the reachability Gramian for an `LTIModel`
at parameter `p`. See `BTReductor` for kwargs.
"""
function reachability_gramian(model::LTIModel, p=nothing; kwargs...)
    reductor = BTReductor(model, p; kwargs...)
    return reachability_gramian(reductor)
end

"""
`Q = observability_gramian(reductor::BTReductor)`

Helper method for forming the observability Gramian.
"""
function observability_gramian(reductor::BTReductor)
    return reductor.L * reductor.L'
end

"""
`Q = observability_gramian(model::LTIModel[, p=nothing; kwargs...])`

Helper method for forming the observability Gramian for an `LTIModel`
at parameter `p`. See `BTReductor` for kwargs.
"""
function observability_gramian(model::LTIModel, p=nothing; kwargs...)
    reductor = BTReductor(model, p; kwargs...)
    return observability_gramian(reductor)
end

"""
`H2_norm(reductor::BTReductor[, withP=true])`

Uses the Gramians of the `reductor` object to compute the
``\\mathcal{H}_2`` norm of `reductor.model`. If `withP==true`,
uses the reachability Grammian, otherwise uses the 
observatility Grammian.
"""
function H2_norm(reductor::BTReductor, withP=true)
    if withP
        # trace(C P Cᵀ) = trace(C R R' C') = || C R ||_F^2
        return norm(reductor.model.C * reductor.R)
    else
        # trace(Bᵀ Q B) = trace(Bᵀ L L' B) = || L' B ||_F^2
        return norm(reductor.L' * reductor.model.B)
    end
end

"""
`H2_norm(model::LTIModel[, p=nothing; withP=true, kwargs...])`

Forms a `BTReductor` and computes the ``\\mathcal{H}_2`` norm of the 
`model`. See `BTReductor` for kwargs.
"""
function H2_norm(model::LTIModel, p=nothing; withP=true, kwargs...)
    reductor = BTReductor(model, p; kwargs...)
    return H2_norm(reductor, withP)
end

"""
`H2_error(m1::LTIModel, m2::LTIModel[, p=nothing; withP=true, kwargs...])`

Forms a `BTReductor` and computes the ``\\mathcal{H}_2`` norm of the 
difference between the models. See `BTReductor` for kwargs.
"""
function H2_error(m1::LTIModel, m2::LTIModel, p=nothing; withP=true, kwargs...)
    diff = m1 - m2
    if !isnothing(p)
        diff(p)
    end
    return H2_norm(diff, p, withP=withP; kwargs...)
end

"""
`Hinf_norm(reductor::BTReductor[, max_iter=100, imag_tol=1e-4, real_tol=1e-4, reltol=1e-2, noise=0])`

Uses the bisection method to compute the `\\mathcal{H}_\\infty`` norm of the
model `reductor.model`. See the following reference.

https://web.stanford.edu/~boyd/papers/pdf/bisection_hinfty.pdf
"""
function Hinf_norm(reductor::BTReductor; max_iter=100, imag_tol=1e-4, real_tol=1e-4, reltol=1e-2, noise=0)
    model = reductor.model
    E = (isa(model.E, UniformScaling) || !issparse(model.E)) ? model.E : Matrix(model.E)
    A = E \ model.A
    B = model.B
    C = model.C
    D = model.D
    sdmax = svd(reductor.model.D).S[1]
    γlb = max(sdmax, reductor.hs[1])
    γub = sdmax + 2 * sum(reductor.hs)
    for iter in 1:max_iter
        γ = (γlb + γub) / 2
        if noise >= 1
            @printf("Iteration %d: Bisecting (%.4e,%.4e)\n", iter, γlb, γub)
        end
        if γub - γlb <= 2 * reltol * γlb
            return γ
        end
        Mγ = begin
            if iszero(D)
                [A      (B*B' ./ γ) ;
                (-C'C ./ γ)  -A']
            else
                [A       0.0.*A ;
                0.0.*A  -A'] .+ 
                [B    spzeros(size(B,1),size(C,1)) ;
                spzeros(size(B,2),size(C,2))  -C'] * 
                [-D γ*I ;
                γ*I -D'] \
                [C    spzeros(size(C,1),size(B,1)) ;
                spzeros(size(B,1),size(C,1))  B']
            end
        end
        Λ = eigen(Matrix(Mγ)).values
        if any(abs.(imag.(Λ)) .> imag_tol .&& abs.(real.(Λ)) .< real_tol)
            γlb = γ
        else
            γub = γ
        end
    end
    if noise >= 1
        println("Did not converge in $max_iter iterations")
    end
    return (γlb + γub) / 2
end

"""
`Hinf_norm(model::LTIModel[, p=nothing; kwargs...])`

Forms a `BTReductor` and computes the ``\\mathcal{H}_\\infty`` norm of the 
`model`. See `BTReductor` and `Hinf_norm` for kwargs.
"""
function Hinf_norm(model::LTIModel, p=nothing; max_iter=100, imag_tol=1e-4, real_tol=1e-4, reltol=1e-2, noise=0, kwargs...)
    reductor = BTReductor(model, p; noise=noise, kwargs...)
    return Hinf_norm(reductor, max_iter=max_iter, imag_tol=imag_tol, 
                     real_tol=real_tol, reltol=reltol, noise=noise)
end

"""
`Hinf_error(m1::LTIModel, m2::LTIModel[, p=nothing; kwargs...])`

Forms a `BTReductor` and computes the ``\\mathcal{H}_\\infty`` norm of the 
difference `m1 - m2`. See `BTReductor` and `Hinf_norm` for kwargs.

https://web.stanford.edu/~boyd/papers/pdf/bisection_hinfty.pdf
"""
function Hinf_error(m1::LTIModel, m2::LTIModel, p=nothing; kwargs...)
    diff = m1 - m2
    if !isnothing(p)
        diff(p)
    end
    return Hinf_norm(diff, p; kwargs...)
end

"""
`form_rom(bt_reductor[, r=-1])`

Uses Petrov-Galerkin on the model to form a ROM
of order `r` (largest possible if `r==-1`). Also,
initializes it to `bt_reductor.p` if not nothing.
"""
function form_rom(bt_reductor::BTReductor, r=-1)
    rom = galerkin_project(bt_reductor.model, bt_reductor.V, bt_reductor.W, WTEVisI=true, r=r)
    if !isnothing(bt_reductor.p)
        rom(bt_reductor.p)
    end
    return rom
end

"""
`lift(bt_reductor, x_r)`

Given a solution array `x_r` to a ROM formed by the
`bt_reductor` lifts the solution(s) to the same dimension of
the FOM.   
"""
function lift(bt_reductor::BTReductor, x_r::AbstractArray)
    r = size(x_r,1)
    V = bt_reductor.V
    return V[:, 1:r] * x_r
end