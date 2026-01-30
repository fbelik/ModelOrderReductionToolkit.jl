"""
`reductor = SISOIRKAReductor(model::LTIModel, r::Int[, p=nothing; init_sigmas=1:r, input_idx=1, output_idx=1, max_iter=50, eps=1e-2, conjugate_tol=1e-6, noise=0])`

`reductor = IRKAReductor(model::LTIModel[, p=nothing; init_sigmas=1:r, max_iter=50, eps=1e-2, conjugate_tol=1e-6, noise=0])`

IRKA reductor object for reducing an `LTIModel`. Must pass in desired reduced order
model dimension `r`. If parameter `p` passed in, model first initialized to parameter value. 
Uses `init_sigmas` to determine initial shifts. `SISOIRKAReductor` produces a ROM for a SISO model or 
focused on a single input (`input_idx`) and output (`output_idx`). Maximum number of IRKA iterations 
determined by `max_iter`. Proceeds until the relative change in the shifts is less than `eps`. 
`conjugate_tol` is used to determine whether or not two complex numbers are complex conjugates of each
other. Use `noise>=1` to print output as the algorithm runs.
"""
struct IRKAReductor
    model::LTIModel
    shifts::Vector{ComplexF64}
    deltas::Vector{Float64}
    V::Matrix{Float64}
    W::Matrix{Float64}
    p
end

function Base.show(io::Core.IO, reductor::IRKAReductor)
    res  = "IRKA Reductor"
    println(io, res)
    print(io, "FOM: ")
    print(io, reductor.model)
    if !isnothing(reductor.p)
        println(io, "")
        print(io, "Initialized to parameter value $(reductor.p)")
    end
end

function compute_relative_set_distance_greedy(a, b, dist_matrix=Matrix{Float64}(undef,length(a),length(b)))
    dist_matrix .= abs.(a .- transpose(b))
    res = 0.0
    for i in eachindex(a)
        dist, j_choice = findmin(view(dist_matrix,i,:))
        dist_matrix[:,j_choice] .= Inf
        dist_matrix[i,j_choice] = dist
        res += (dist / max(1e-12,abs(a[i]))) ^ 2
    end
    return sqrt(res)
end

function SISOIRKAReductor(model::LTIModel, r::Int, p=nothing; init_sigmas=1:r, input_idx=1, output_idx=1, max_iter=50, eps=1e-2, conjugate_tol=1e-6, noise=0)
    if !isnothing(p)
        model(p)
    elseif is_parameterized(model) && noise >= 1
        println("No parameter inputted, make sure the LTI model is not parameterized or already initialized to a parameter")
    end
    if length(init_sigmas) != r
        error("length(init_sigmas) must equal r=$r")
    end
    @assert isreal(model.A)
    σs = sort(init_sigmas .+ 0.0im, by=(x -> (real(x),imag(x))))
    deltas = Float64[]
    V = Matrix{Float64}(undef, size(model.A,2), r)
    W = Matrix{Float64}(undef, size(model.A,2), r)
    b = view(model.B,:,input_idx)
    c = view(model.C,output_idx,:)
    dist_matrix = zeros(r,r)
    for iter in 1:max_iter
        # Form V and W
        i = 1
        while i <= r
            σ = σs[i]
            M = factorize(σ*model.E - model.A)
            v = M \ b
            w = (M') \ c
            if i <= r-1 && abs(σ - σs[i+1]') < conjugate_tol
                v1 = real.(v); orthonormalize_mgs2!(v1, view(V,:,1:(i-1)))
                w1 = real.(w); orthonormalize_mgs2!(w1, view(W,:,1:(i-1)))
                V[:,i] .= v1
                W[:,i] .= w1
                v2 = imag.(v); orthonormalize_mgs2!(v2, view(V,:,1:i))
                w2 = imag.(w); orthonormalize_mgs2!(w2, view(W,:,1:i))
                V[:,i+1] .= v2
                W[:,i+1] .= w2
                i += 2
            else
                v1 = real.(v); orthonormalize_mgs2!(v1, view(V,:,1:(i-1)))
                V[:,i] .= v1
                w1 = real.(w); orthonormalize_mgs2!(w1, view(W,:,1:(i-1)))
                W[:,i] .= w1
                i += 1
            end
        end
        # Update W -> W ((W' E V)⁻¹)' such that W' E V = I
        W .= W / ((W' * model.E * V)')
        # Form Aᵣ and compute spectrum
        Ar = W' * model.A * V
        negλs = sort(-1 .* eigen(Ar).values, by=(x -> (real(x),imag(x))))
        # Break condition
        delta = compute_relative_set_distance_greedy(σs, negλs, dist_matrix)
        if noise >= 1
            @printf("(%d): Relative error %.4f%%\n", iter, 100 * delta)
        end
        push!(deltas, delta)
        if delta <= eps 
            break
        end
        σs .= negλs
    end
    return IRKAReductor(model, σs, deltas, V, W, p)
end

function IRKAReductor(model::LTIModel, r::Int, p=nothing; init_sigmas=1:r, max_iter=50, eps=1e-2, conjugate_tol=1e-6, noise=0)
    if !isnothing(p)
        model(p)
    elseif is_parameterized(model) && noise >= 1
        println("No parameter inputted, make sure the LTI model is not parameterized or already initialized to a parameter")
    end
    if length(init_sigmas) != r
        error("length(init_sigmas) must equal r=$r")
    end
    @assert isreal(model.A)
    σs = sort(init_sigmas .+ 0.0im, by=(x -> (real(x),imag(x))))
    deltas = Float64[]
    n = size(model.A,2)
    V = Matrix{Float64}(undef, n, r)
    W = Matrix{Float64}(undef, n, r)
    B = model.B; n_in = size(B, 2)
    C = model.C; n_out = size(C, 1)
    bs = zeros(ComplexF64, n_in, r)
    cs = zeros(ComplexF64, n_out, r)
    i = 1
    while i <= r
        if i <= r-1 && abs(σs[i] - σs[i+1]') < conjugate_tol
            bs[:,i] .= rand(ComplexF64, n_in)
            bs[:,i+1] .= adjoint.(bs[:,i])
            cs[:,i] .= rand(ComplexF64, n_out)
            cs[:,i+1] .= adjoint.(cs[:,i])
            i += 2
        else # Conjugate pairs
            bs[:,i] .= rand(Float64, n_in)
            cs[:,i] .= rand(Float64, n_out)
            i += 1
        end
    end
    dist_matrix = zeros(r,r)
    for iter in 1:max_iter
        # Form V and W
        i = 1
        while i <= r
            σ = σs[i]
            M = factorize(σ*model.E - model.A)
            v = M \ (B * view(bs, :, i))
            w = (M') \ (C' * view(cs, :, i))
            if i <= r-1 && abs(σ - σs[i+1]') < conjugate_tol
                v1 = real.(v); orthonormalize_mgs2!(v1, view(V,:,1:(i-1)))
                w1 = real.(w); orthonormalize_mgs2!(w1, view(W,:,1:(i-1)))
                V[:,i] .= v1
                W[:,i] .= w1
                v2 = imag.(v); orthonormalize_mgs2!(v2, view(V,:,1:i))
                w2 = imag.(w); orthonormalize_mgs2!(w2, view(W,:,1:i))
                V[:,i+1] .= v2
                W[:,i+1] .= w2
                i += 2
            else
                v1 = real.(v); orthonormalize_mgs2!(v1, view(V,:,1:(i-1)))
                V[:,i] .= v1
                w1 = real.(w); orthonormalize_mgs2!(w1, view(W,:,1:(i-1)))
                W[:,i] .= w1
                i += 1
            end
        end
        # Update W -> W ((W' E V)⁻¹)' such that W' E V = I
        W .= W / ((W' * model.E * V)')
        # Form Aᵣ and compute spectrum
        Ar = W' * model.A * V
        Br = W' * B
        Cr = C * V
        λ, X = eigen(Ar)
        bs .= (Br') / X
        cs .= Cr * X
        negλs = sort(-1 .* λ, by=(x -> (real(x),imag(x))))
        # Break condition
        delta = compute_relative_set_distance_greedy(σs, negλs, dist_matrix)
        if noise >= 1
            @printf("(%d): Relative error %.4f%%\n", iter, 100 * delta)
        end
        push!(deltas, delta)
        if delta <= eps 
            break
        end
        σs .= negλs
    end
    return IRKAReductor(model, σs, deltas, V, W, p)
end

"""
`form_rom(irka_reductor[, r=-1])`

Uses Petrov-Galerkin on the model to form a ROM
of order `r` (largest possible if `r=-1`). Also,
initializes it to `irka_reductor.p` if not nothing.
"""
function form_rom(irka_reductor::IRKAReductor, r=-1)
    rom = galerkin_project(irka_reductor.model, irka_reductor.V, irka_reductor.W, WTEVisI=true, r=r)
    if !isnothing(irka_reductor.p)
        rom(irka_reductor.p)
    end
    return rom
end

"""
`lift(irka_reductor, x_r)`

Given a solution array `x_r` to a ROM formed by the
`irka_reductor` lifts the solution(s) to the same dimension of
the FOM.   
"""
function lift(irka_reductor::IRKAReductor, x_r::AbstractArray)
    r = size(x_r,1)
    V = irka_reductor.V
    if r < size(V, 2)
        return V[:, 1:r] * x_r
    else
        return V * x_r
    end
end