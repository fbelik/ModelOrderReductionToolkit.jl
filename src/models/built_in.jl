"""
`model = PoissonModel([; Nx=999, P=3])`

Uses finite differences on the following PDE
to generate a `LinearModel` with parameter 
dependence. Change discretization with `Nx`, 
and change number of parameters, and hence number
of affine terms, with `P`.

`- ∂_x (κ(x, p) ∂_x u(x, p)) = f(x)`

`u(0,p) = 0; u(1,p) = p[1]`

`κ(x,p) = (1.05 - (1/2)^(P-1)) + 0.5 p[2] sin(2πx) + 0.25 p[3] sin(4πx) + ... + (1/2)^(P-1) p[P] sin(2π * P * x)`

`f(x,p) = (0.25 .< x .< 0.75) .* 10.0`

`length(p) = P; 0 ≤ p[i] ≤ 1`
"""
function PoissonModel(; Nx=999, P=3)
    # Parameter dependent diffusion coefficient
    κ_i(i,x) = i==1 ? (1.05 - (1/2) ^ (P-1)) : (0.5^(i-1) .* sin.(2π * i * x))
    κ(x,p) = κ_i(1,x) + sum([p[i] * κ_i(i,x) for i in 2:P])
    # Parameter dependent source term
    f(x) = (0.25 .< x .< 0.75) .* 10.0
    # Space setup
    xs = range(0,1,Nx+2)[2:end-1]
    dx = xs[2] - xs[1]
    xhalfs = (range(0,1,Nx+2)[1:end-1] .+ range(0,1,Nx+2)[2:end]) ./ 2
    N = length(xs)

    Ais = AbstractMatrix[]
    for i in 1:P
        A = spzeros(N,N)
        for j in 1:N
            A[j,j]   = (κ_i(i, xhalfs[j]) + κ_i(i, xhalfs[j+1])) / dx^2
            if j<N
                A[j,j+1] = -1 * κ_i(i, xhalfs[j+1]) / dx^2
            end
            if j>1
                A[j,j-1] = -1 * κ_i(i, xhalfs[j]) / dx^2
            end
        end
        push!(Ais, A)
    end

    function makeθAi(p,i)
        if i == 1
            return 1.0
        else
            return p[i]
        end
    end

    bis = Vector[]
    for i in 1:P
        b = spzeros(length(xs))
        # b[1] += uleft * κ_i(i, xhalfs[1]) / dx^2 
        b[end] += 1.0 * κ_i(i, xhalfs[end]) / dx^2
        push!(bis, b)
    end
    push!(bis, f(xs))

    function makeθbi(p,i)
        if i == 1
            return p[1]
        elseif i == (P+1)
            return 1.0
        else
            return p[1] * p[i]
        end
    end

    Ap = APArray(Ais, makeθAi)
    bp = APArray(bis, makeθbi)

    return LinearModel(Ap, bp)
end

"""
`model = PenzlModel()`

Generates the standard Penzl `LTIModel` with
one input, one output, and `ns=1006` dimension state variable.
"""
function PenzlModel(ns::Int=1006)
    matrices = [
        [-1 100; -100 -1],
        [-1 200; -200 -1],
        [-1 400; -400 -1],
        spdiagm(-1:-1:(-1*(ns-6))),
    ]
    N = sum(size(matrix,1) for matrix in matrices)
    A = spzeros(N,N)
    idx=1
    for i in eachindex(matrices)
        matrix = matrices[i]
        n = size(matrix,1)
        A[idx:idx+n-1,idx:idx+n-1] .= matrix
        idx += n
    end
    B = zeros(N,1)
    B[1:6] .= 10
    B[7:end] .= 1
    C = B'
    return LTIModel(A, B, C)
end

"""
`model = MISOPenzlModel()`

Generates an `LTIModel` with three inputs, one output, 
and a state of dimension `ns=1006`. Same structure as the 
Penzl model except the `B` matrix is changed to

```
1006×3 Matrix{Float64}:
 10.0   0.0   0.0
 10.0   0.0   0.0
  0.0  10.0   0.0
  0.0  10.0   0.0
  0.0   0.0  10.0
  0.0   0.0  10.0
  1.0   1.0   1.0
  ⋮
  1.0   1.0   1.0
  1.0   1.0   1.0
  1.0   1.0   1.0
  1.0   1.0   1.0
  1.0   1.0   1.0
  1.0   1.0   1.0
  1.0   1.0   1.0
```
"""
function MISOPenzlModel(ns::Int=1006)
    matrices = [
        [-1 100; -100 -1],
        [-1 200; -200 -1],
        [-1 400; -400 -1],
        spdiagm(-1:-1:(-1*(ns-6))),
    ]
    N = sum(size(matrix,1) for matrix in matrices)
    A = spzeros(N,N)
    idx=1
    for i in eachindex(matrices)
        matrix = matrices[i]
        n = size(matrix,1)
        A[idx:idx+n-1,idx:idx+n-1] .= matrix
        idx += n
    end
    B = ones(N,3)
    B[1:2, 1] .= 10
    B[3:6, 1] .= 0
    B[1:2, 2] .= 0
    B[3:4, 2] .= 10
    B[5:6, 2] .= 0
    B[1:4, 3] .= 0
    B[5:6, 3] .= 10
    C = ones(1,N)
    C[1, 1:6] .= 10.0
    return LTIModel(A, B, C)
end

"""
`model = ParameterizedPenzlModel()`

Generates an `LTIModel` with one input, one output, 
and a state of dimension `ns=1006`. Same structure as the 
Penzl model, expect depends on a parameter vector of
length 3 which shift the poles along the complex axis.
Instantiate to a parameter vector by calling
`model([p1,p2,p3])`.
"""
function ParameterizedPenzlModel(ns::Int=1006)
    allmats = [
        [
            [-1 100; -100 -1],
            [-1 200; -200 -1],
            [-1 400; -400 -1],
            spdiagm(-1:-1:(-1*(ns-6))),
        ],
        [
            [0 1; -1 0],
            [0 0; 0 0],
            [0 0; 0 0],
            0 .* spdiagm(-1:-1:(-1*(ns-6))),
        ],
        [
            [0 0; 0 0],
            [0 1; -1 0],
            [0 0; 0 0],
            0 .* spdiagm(-1:-1:(-1*(ns-6))),
        ],
        [
            [0 0; 0 0],
            [0 0; 0 0],
            [0 1; -1 0],
            0 .* spdiagm(-1:-1:(-1*(ns-6))),
        ]
    ]
    N = sum(size(matrix,1) for matrix in allmats[1])
    Ais = []
    for matrices in allmats
        A = spzeros(N,N)
        idx=1
        for i in eachindex(matrices)
            matrix = matrices[i]
            n = size(matrix,1)
            A[idx:idx+n-1,idx:idx+n-1] .= matrix
            idx += n
        end
        push!(Ais, A)
    end
    makeθAi = (p, i) -> begin
        if (i == 1)
            return 1.0
        else
            return p[i-1]
        end
    end
    Ap = APArray(Ais, makeθAi)
    B = zeros(N,1)
    B[1:6] .= 10
    B[7:end] .= 1
    C = B'
    return LTIModel(Ap, B, C)
end