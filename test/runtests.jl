using ModelOrderReductionToolkit
using Test
using SparseArrays
using LinearAlgebra

@testset "Linear Greedy RBM Methods" begin
    # Initialize toy problem
    model = PoissonModel()
    params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
    # Test SCM SPD and Noncoercive procedures 
    Ma = 15; Mp = 15; ϵ_SCM = 1e-2;
    scm = initialize_SCM_SPD(params, model.Ap, Ma, Mp, ϵ_SCM, noise=1);
    @test length(scm.σ_UBs) <= 25
    errs = Float64[]
    for p in params
        (lb,ub) = find_sigma_bounds(scm, p)
        push!(errs, (ub - lb) / ub)
    end
    @test maximum(errs) < ϵ_SCM
    scm2 = initialize_SCM_Noncoercive(params, model.Ap, Ma, Mp, ϵ_SCM, noise=1);
    @test length(scm2.σ_UBs) <= 70
    errs = Float64[]
    for p in params
        (lb,ub) = find_sigma_bounds(scm2, p)
        push!(errs, (ub - lb) / ub)
    end
    @test maximum(errs) < ϵ_SCM
    # Test GreedyRBAffineLinear on different residual computations
    ϵ_greedy = 1e-2
    error_estimator = StabilityResidualErrorEstimator(model, scm)
    reductor = WGReductor(model, error_estimator)
    add_to_rb!(reductor, params, 50, eps=ϵ_greedy)
    @test size(reductor.V, 2) <= 40
    # Test with stability radial basis function
    # Sample 1/3 of params
    np = length(params)
    indices = floor.(Int,range(1,np,floor(Int,np/3)))
    params_subset = params[indices]
    makeA = (p) -> begin
        formArray!(model.Ap, model.A_alloc, p)
        model.A_alloc
    end
    stability_rbf = min_sigma_rbf(params_subset,makeA)
    error_estimator = StabilityResidualErrorEstimator(model, stability_rbf)
    reductor = WGReductor(model, error_estimator)
    add_to_rb!(reductor, params, 31, eps=ϵ_greedy)
    @test size(reductor.V, 2) <= 30
end