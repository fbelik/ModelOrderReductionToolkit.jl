using ModelOrderReductionToolkit
using Test
using SparseArrays
using LinearAlgebra

@testset "Linear Greedy RBM Methods" begin
    # Initialize toy problem
    begin
        uleft = 0.0
        uright = 1.0
        # Parameter dependent diffusion coefficient
        P = 2
        κ_i(i,x) = 1.1 .+ sin.(2π * i * x)
        κ(x,p) = sum([p[i] * κ_i(i,x) for i in 1:P])
        # Parameter dependent source term
        f_i(i,x) = i == 1 ? (x .> 0.5) .* 10.0 : 20 .* sin.(2π * (i-1) * x)
        f(x,p) = f_i(1,x) .+ sum([p[i-1] * f_i(i,x) for i in 2:P+1])
        # Space setup
        h = 1e-2
        xs = Vector((0:h:1)[2:end-1])
        xhalfs = ((0:h:1)[1:end-1] .+ (0:h:1)[2:end]) ./ 2
        N = length(xs)
        # Helper to generate random parameter vector (in [0,2])
        randP() = 0.1 .+ 2 .* rand(P);
        makeAi = (i) -> begin
            A = spzeros(N,N)
            for j in 1:N
                A[j,j]   = (κ_i(i, xhalfs[j]) + κ_i(i, xhalfs[j+1])) / h^2
                if j<N
                    A[j,j+1] = -1 * κ_i(i, xhalfs[j+1]) / h^2
                end
                if j>1
                    A[j,j-1] = -1 * κ_i(i, xhalfs[j]) / h^2
                end
            end
            return A
        end
        function makebi(i)
            b = f_i(i,xs)
            if i > 1
                b[1] += uleft * κ_i(i, xhalfs[1]) / h^2 
                b[end] += uright * κ_i(i, xhalfs[end]) / h^2
            end
            return b
        end
        params = []
        for p1 in range(0.1,2.1,10)
            for p2 in range(0.1,2.1,10)
                push!(params, [p1,p2])
            end
        end
        Ais = []
        for i in 1:P
            push!(Ais, makeAi(i))
        end
        bis = []
        for i in 1:P+1
            push!(bis, makebi(i))
        end
        function makeθAi(p,i)
            return p[i]
        end
        function makeθbi(p,i)
            if i == 1
                return 1.0
            else
                return p[i-1]
            end
        end
        makeA = (p) -> begin
            A = spzeros(size(Ais[1]))
            for i in eachindex(Ais)
                A .+= makeθAi(p,i) .* Ais[i]
            end
            return A
        end
        makeb = (p) -> begin
            b = zeros(size(bis[1]))
            for i in eachindex(bis)
                b .+= makeθbi(p,i) .* bis[i]
            end
            return b
        end
        truth_sol(p) = makeA(p) \ makeb(p)
    end
    # Test SCM SPD and Noncoercive procedures 
    Ma = 15; Mp = 15; ϵ_SCM = 1e-2;
    scm = initialize_SCM_SPD(params, Ais, makeθAi, Ma, Mp, ϵ_SCM, noise=1);
    @test length(scm.σ_UBs) <= 15
    errs = Float64[]
    for p in params
        (lb,ub) = find_sigma_bounds(scm, p)
        push!(errs, (ub - lb) / ub)
    end
    @test maximum(errs) < ϵ_SCM
    scm2 = initialize_SCM_Noncoercive(params, Ais, makeθAi, Ma, Mp, ϵ_SCM, noise=1);
    @test length(scm2.σ_UBs) <= 50
    errs = Float64[]
    for p in params
        (lb,ub) = find_sigma_bounds(scm2, p)
        push!(errs, (ub - lb) / ub)
    end
    @test maximum(errs) < ϵ_SCM
    # Test GreedyRBAffineLinear on different residual computations
    ϵ_greedy = 1e-2
    for i in 0:2
        greedy_sol = GreedyRBAffineLinear(params, Ais, makeθAi, bis, makeθbi, scm, ϵ_greedy, noise=1)
        @test length(greedy_sol.params_greedy) <= 30
    end
    # Test with stability radial basis function
    # Sample 1/3 of params
    np = length(params)
    indices = floor.(Int,range(1,np,floor(Int,np/3)))
    params_subset = params[indices]
    stability_rbf = min_sigma_rbf(params_subset,makeA)
    greedy_sol_rbf = GreedyRBAffineLinear(params, Ais, makeθAi, bis, makeθbi, stability_rbf, ϵ_greedy, noise=1)
    @test length(greedy_sol_rbf.params_greedy) <= 30
end