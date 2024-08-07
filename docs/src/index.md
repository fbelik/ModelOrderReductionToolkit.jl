# ModelOrderReductionToolkit.jl Models and Reductors

## Stationary Models

A `model <: StationaryModel{NOUT}` is a map from parameter space to a vector space. When calling `model(p, i=1)` for `i=1,...,NOUT`, must return a vector of length `output_length(model)` and of eltype `output_type(model)`. 

### Linear Model

A model of the form
```math
A(p) x(p) = b(p).
```
```@docs
LinearModel
```
For an example, see `PoissonModel()`.

### Linear Matrix Model

A model of the form
```math
A(p) X(p) = B(p).
```
```@docs
LinearMatrixModel
```
For an example, see `to_frequency_domain(PenzlModel())`.

## Stationary Model Reductors

Reduced basis methods for stationary models typically require the models to have implemented `galerkin_project(model, V[, W=V], r=-1)` and `galerkin_add!(rom, fom, v, Vold[, w=v, Wold=Vold])`.

### POD Reductor

Proper orthogonal decomposition which solves the full order model at each parameter value, computes an SVD of the snapshot matrix, and allows for projection onto the resulting left singular vectors.

```@docs
PODReductor
```

```julia
model = PoissonModel()
params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
reductor = PODReductor(model)
add_to_rb!(reductor, params) # or add_to_rb!(reductor, snapshots) if snapshot matrix already formed
rom = form_rom(reductor, 10)
```

### SG Reductor

Strong greedy method which solves the full order model at each parameter value, computes a column-pivoted QR decomposition of the snapshot matrix, and allows for projection onto the resulting orthonormalized snapshots.

```@docs
SGReductor
```

```julia
model = PoissonModel()
params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
reductor = SGReductor(model)
add_to_rb!(reductor, params) # or add_to_rb!(reductor, snapshots) if snapshot matrix already formed
rom = form_rom(reductor, 10)
```

### WG Reductor

Weak greedy method which utilizes an `ErrorEstimator` to attempt to choose the parameter value (and index) for which the error is the worst for the given RB. Then, forms that full order solution and continues.
```@docs
WGReductor
```

```julia
model = PoissonModel()
params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
estimator = StabilityResidualErrorEstimator(model, params, coercive=true)
reductor = WGReductor(model, estimator)
add_to_rb!(reductor, params, 10)
rom = form_rom(reductor, 10)
```

## Nonstationary Models

A `model <: NonstationaryModel` is a map from parameter space to a set of ODEs. Calling `model(p)` initializes the model to the given parameter value. ODE outputs are vector of length `output_length(model)` and of eltype `output_type(model)`. Can convert into an `ODEProblem` by `to_ode_problem(model[, x0, tspan, p])`.

### LTI Model

A model of the form
```math
E(p) x'(t;p) = A(p) x(t;p) + B(p) u(t)\\
y(t;p) = C(p) x(t;p) + D(p) u(t)
```

```@docs
LTIModel
```

Also implements `to_ss(model[, p=nothing])` to convert a model to a `ControlSystems.jl` state space, and `to_dss(model[, p=nothing])` to convert to a `DescriptorSystems.jl`. Also implements `to_frequency_domain(model)` which returns a `LinearMatrixModel` to solve for the state variable `x` in the frequency domain. Also has `galerkin_project(model, V[, W=V; r=-1])` implemented for forming reduced order models. To cast to an ODE problem for a given input `u(x)`, call `to_ode_problem(model[, x0=0.0, tspan=(0,1), p=nothing, u])

The state of the art method for reducing a non-parameterized LTI problem is through balanced truncation. This can currently be done in `DescriptorSystems.jl`. (Goal of implementing a faster iterative solver here)
```julia
model = PenzlModel()
sys = to_dss(model)
sys_r, hs = gbalmr(sys; balance=true) # Returns a reduced system and HSVs
rom = LTIModel(sys_r)
```

For reducing a parameterized LTI problem in a reduced basis setting, one option is to create a (complex) basis in the frequency domain.

```julia
model = ParameterizedPenzlModel()
freq_model = to_frequency_domain(model)
reductor = PODReductor(freq_model)
params = [[ω,p1,p2,p3] for ω in range(-500,500,21) for p1 in range(-25,25,6) for p2 in range(-25,25,6) for p3 in range(-25,25,6)]
reductor = PODReductor(freq_model)
add_to_rb!(reductor, params)
rom = galerkin_project(model, Matrix(reductor.V[:,1:20])) # Faster when converting from VOV to Matrix
```