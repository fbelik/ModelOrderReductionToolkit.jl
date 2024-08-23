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
        bodeerr = abs.(bode(model, omegas, p, first=true) .- bode(rom, omegas, p, first=true))
        println("Bode error for rb method at p=$p - $(maximum(bodeerr))")
        @test maximum(bodeerr) < ERR_TOL
    end
end