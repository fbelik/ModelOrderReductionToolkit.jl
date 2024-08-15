using ModelOrderReductionToolkit
using Test
using SparseArrays
using LinearAlgebra

@testset "LinearModel" begin
    model = PoissonModel()
    r = 25; ERR_TOL = 1e-2
    params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
    # POD Reductor
    pod_reductor = PODReductor(model)
    add_to_rb!(pod_reductor, params)
    rom = form_rom(pod_reductor, r)
    errs = Float64[]
    for (i,p) in enumerate(params)
        push!(errs, norm(lift(pod_reductor, rom(p)) .- pod_reductor.snapshots[:,i]))
    end
    @test maximum(errs) < ERR_TOL
    # SG Reductor
    sg_reductor = SGReductor(model)
    add_to_rb!(sg_reductor, pod_reductor.snapshots)
    rom = form_rom(sg_reductor, r)
    errs = Float64[]
    for (i,p) in enumerate(params)
        push!(errs, norm(lift(sg_reductor, rom(p)) .- sg_reductor.snapshots[:,i]))
    end
    @test maximum(errs) < ERR_TOL
    # WG Reductor
    SCM_TOL = 1e-1
    for c in [true,false]
        error_estimator = StabilityResidualErrorEstimator(model, params, coercive=c)
        # Test that SCM has sufficiently small ϵ
        errs = Float64[]
        for p in params
            (lb,ub) = find_sigma_bounds(error_estimator.stability_estimator, p)
            push!(errs, (ub - lb) / ub)
        end
        @test maximum(errs) < SCM_TOL
        wg_reductor = WGReductor(model, error_estimator)
        add_to_rb!(wg_reductor, params, r, eps=0)
        rom = form_rom(wg_reductor, r)
        errs = Float64[]
        for (i,p) in enumerate(params)
            push!(errs, norm(lift(wg_reductor, rom(p)) .- pod_reductor.snapshots[:,i]))
        end
        @test maximum(errs) < ERR_TOL
    end
    makeA = (p) -> begin
        formArray!(model.Ap, model.A_alloc, p)
        model.A_alloc
    end
    stability_rbf = min_sigma_rbf(params,makeA)
    error_estimator = StabilityResidualErrorEstimator(model, stability_rbf)
    wg_reductor = WGReductor(model, error_estimator)
    add_to_rb!(wg_reductor, params, r, eps=0)
    rom = form_rom(wg_reductor, r)
    errs = Float64[]
    for (i,p) in enumerate(params)
        push!(errs, norm(lift(wg_reductor, rom(p)) .- pod_reductor.snapshots[:,i]))
    end
    @test maximum(errs) < ERR_TOL
end

@testset "LTIModel" begin
    model = ParameterizedPenzlModel()
    r = 25; ERR_TOL = 1e-2
    p0 = zeros(3)
    # Balanced truncation
    bt_reductor = BTReductor(model, p0)
    rom = form_rom(bt_reductor, r)
    omegas = range(-500,500,1001)
    bodeerr = abs.(bode(model, omegas, first=true) .- bode(rom, omegas, first=true))
    println("Bode error for bt method - $(maximum(bodeerr))")
    @test maximum(bodeerr) < ERR_TOL
    # RB Method
    freq_model = to_frequency_domain(model)
    params = [[ω,i,j,k] for ω in range(-500,500,21) for i in range(-40,40,5) for j in range(-40,40,5) for k in range(-40,40,5)]
    rb_reductor = PODReductor(freq_model)
    add_to_rb!(rb_reductor, params)
    rom = galerkin_project(model, Matrix(rb_reductor.V[:,1:r]))
    omegas = range(-500,500,1001)
    for p in [[0,0,0],[0,0,50],[40,-40,0],[10,15,-15]]
        rom(p); model(p);
        bodeerr = abs.(bode(model, omegas, first=true) .- bode(rom, omegas, first=true))
        println("Bode error for rb method at p=$p - $(maximum(bodeerr))")
        @test maximum(bodeerr) < ERR_TOL
    end
end
# @testset "Linear Greedy RBM Methods" begin
#     # Initialize toy problem
#     model = PoissonModel()
#     params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
#     # Test SCM SPD and Noncoercive procedures 
#     Ma = 15; Mp = 15; ϵ_SCM = 1e-2;
#     scm = initialize_SCM_SPD(params, model.Ap, Ma, Mp, ϵ_SCM, noise=1);
#     @test length(scm.σ_UBs) <= 25
#     errs = Float64[]
#     for p in params
#         (lb,ub) = find_sigma_bounds(scm, p)
#         push!(errs, (ub - lb) / ub)
#     end
#     @test maximum(errs) < ϵ_SCM
#     scm2 = initialize_SCM_Noncoercive(params, model.Ap, Ma, Mp, ϵ_SCM, noise=1);
#     @test length(scm2.σ_UBs) <= 70
#     errs = Float64[]
#     for p in params
#         (lb,ub) = find_sigma_bounds(scm2, p)
#         push!(errs, (ub - lb) / ub)
#     end
#     @test maximum(errs) < ϵ_SCM
#     # Test GreedyRBAffineLinear on different residual computations
#     ϵ_greedy = 1e-2
#     error_estimator = StabilityResidualErrorEstimator(model, scm)
#     reductor = WGReductor(model, error_estimator)
#     add_to_rb!(reductor, params, 50, eps=ϵ_greedy)
#     @test size(reductor.V, 2) <= 40
#     # Test with stability radial basis function
#     # Sample 1/3 of params
#     np = length(params)
#     indices = floor.(Int,range(1,np,floor(Int,np/3)))
#     params_subset = params[indices]
#     makeA = (p) -> begin
#         formArray!(model.Ap, model.A_alloc, p)
#         model.A_alloc
#     end
#     stability_rbf = min_sigma_rbf(params_subset,makeA)
#     error_estimator = StabilityResidualErrorEstimator(model, stability_rbf)
#     reductor = WGReductor(model, error_estimator)
#     add_to_rb!(reductor, params, 31, eps=ϵ_greedy)
#     @test size(reductor.V, 2) <= 30
# end