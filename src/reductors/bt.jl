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
            E = model.E
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
`P = reachability_gramian(reductor)`

Helper method for forming the reachability Gramian.
"""
function reachability_gramian(reductor::BTReductor)
    return reductor.R * reductor.R'
end

"""
`Q = observability_gramian(reductor)`

Helper method for forming the observability Gramian.
"""
function observability_gramian(reductor::BTReductor)
    return reductor.L * reductor.L'
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

Given a vector solution `x_r` to a ROM formed by the
`pod_reductor`, which is of smaller dimension than outputs
of the FOM, lifts the solution to the same dimension of
the FOM. 
"""
function lift(bt_reductor::BTReductor, x_r::AbstractVector)
    r = length(x_r)
    V = bt_reductor.V
    N, M = size(V)
    return view(V, 1:N, 1:r) * x_r
end