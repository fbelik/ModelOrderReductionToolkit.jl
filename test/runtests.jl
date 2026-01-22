using ModelOrderReductionToolkit
using Test
using SparseArrays
using LinearAlgebra

@testset "LinearModel" begin
    model = PoissonModel(Nx=100)
    r = 20; ERR_TOL = 1e-2
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
    SCM_EPS = 0.5
    # Test on all kinds of SCM
    scms = [SCM(model.Ap, params, SCM_EPS, coercive=true),
            SCM(model.Ap, params, SCM_EPS, coercive=false),
            ANLSCM(model.Ap, params, SCM_EPS),
            NNSCM(model.Ap, params, SCM_EPS)]
    # Test that each kind of SCM didn't do too much work 
    @test length(keys(scms[1].UBs)) < length(params) / 2
    @test length(keys(scms[2].UBs)) < length(params) / 2
    @test length(keys(scms[3].UBs)) < length(params) / 2
    @test length(keys(scms[4].UBs)) < length(params) / 2
    for scm in scms
        # Test that SCM has sufficiently small ϵ but not negative
        relerrs = scm.(params, which=:E)
        @test maximum(relerrs) <= SCM_EPS
        @test minimum(relerrs) > -1e-2
        error_estimator = StabilityResidualErrorEstimator(model, scm)
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
    model = ParameterizedPenzlModel(100)
    r = 20; ERR_TOL = 1e-2
    p0 = zeros(3)
    # Balanced truncation
    bt_reductor = BTReductor(model, p0)
    rom = form_rom(bt_reductor, r)
    omegas = logrange(1e-2,1e3,100)
    bodeerr = abs.(bode(model, omegas, first=true) .- bode(rom, omegas, first=true))
    @test maximum(bodeerr) < ERR_TOL
    # IRKA
    irka_reductor = IRKAReductor(model, r, p0)
    rom = form_rom(irka_reductor)
    bodeerr = abs.(bode(model, omegas, first=true) .- bode(rom, omegas, first=true))
    @test maximum(bodeerr) < ERR_TOL
    # SISO IRKA
    irka_reductor = SISOIRKAReductor(model, r, p0)
    rom = form_rom(irka_reductor, r)
    omegas = range(-500,500,1001)
    bodeerr = abs.(bode(model, omegas, first=true) .- bode(rom, omegas, first=true))
    @test maximum(bodeerr) < ERR_TOL
    # POD RB Method
    freq_model = to_frequency_domain(model)
    params = [[ω,i,j,k] for ω in 10.0 .^ range(-2,3,50) for i in range(-40,40,5) for j in range(-40,40,5) for k in range(-40,40,5)]
    pod_reductor = PODReductor(freq_model, force_rb_real=true)
    add_to_rb!(pod_reductor, params)
    rom = galerkin_project(model, Matrix(pod_reductor.V[:,1:r]))
    for p in [[0,0,0],[0,0,50],[40,-40,0],[10,15,-15]]
        bodeerr = abs.(bode(model, omegas, p, first=true) .- bode(rom, omegas, p, first=true))
        @test maximum(bodeerr) < ERR_TOL
    end
    # SG RB Method
    sg_reductor = SGReductor(freq_model, force_rb_real=true)
    add_to_rb!(sg_reductor, pod_reductor.snapshots)
    rom = galerkin_project(model, Matrix(sg_reductor.V[:,1:r]))
    for p in [[0,0,0],[0,0,50],[40,-40,0],[10,15,-15]]
        bodeerr = abs.(bode(model, omegas, p, first=true) .- bode(rom, omegas, p, first=true))
        @test maximum(bodeerr) < ERR_TOL
    end
    # WG RB Method
    estimator = StabilityResidualErrorEstimator(freq_model, p -> 1.0)
    wg_reductor = WGReductor(freq_model, estimator, force_rb_real=true)
    add_to_rb!(wg_reductor, params, div(r,2))
    rom = galerkin_project(model, Matrix(sg_reductor.V[:,1:r]))
    omegas = 10.0 .^ range(-2,3,1000)
    for p in [[0,0,0],[0,0,50],[40,-40,0],[10,15,-15]]
        bodeerr = abs.(bode(model, omegas, p, first=true) .- bode(rom, omegas, p, first=true))
        @test maximum(bodeerr) < ERR_TOL
    end
end