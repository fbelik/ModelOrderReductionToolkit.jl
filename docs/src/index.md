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
scm = SCM(model.Ap, params, coercive=true)
estimator = StabilityResidualErrorEstimator(model, scm)
reductor = WGReductor(model, estimator)
add_to_rb!(reductor, params, 10)
rom = form_rom(reductor, 10)
```
See sources 1-3 on more information on successive constraint methods, residual computations, and methodology on weak greedy reduced basis methods.

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

Also implements `to_ss(model[, p=nothing])` to convert a model to a `ControlSystems.jl` state space, and `to_dss(model[, p=nothing])` to convert to a `DescriptorSystems.jl`. Also implements `to_frequency_domain(model)` which returns a `LinearMatrixModel` to solve for the state variable `x` in the frequency domain. Also has `galerkin_project(model, V[, W=V; r=-1])` implemented for forming reduced order models. To cast to an ODE problem for a given input `u(x)`, call `to_ode_problem(model[, x0=0.0, tspan=(0,1), p=nothing, u])`.

### BT Reductor

The state of the art method for reducing a non-parameterized LTI problem is through balanced truncation. This can be performed with a `BTReductor` object.
```@docs
BTReductor
```

After forming a `BTReductor` object on an `LTIModel`, can obtain the system Gramians through `reachability_gramian(reductor)` and `observability_gramian(reductor)`. As with other reductors, has `form_rom` and `lift` implemented.

```julia
model = PenzlModel()
reductor = BTReductor(model)
rom = form_rom(reductor, 20)
```

See source 4 for more information on the iterative method for solving Lyapunov equations used by `BTReductor` when `iterative==true`, and see `MatrixEquations.jl` for the non-sparse Lyapunov solver. See source 5 for the Penzl model example and for more information on truncation of LTI systems.

### RB Reduction

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

### References:
1. D.B.P. Huynh, G. Rozza, S. Sen, A.T. Patera. A successive constraint linear optimization method for lower bounds of parametric coercivity and inf–sup stability constants. Comptes Rendus Mathematique. Volume 345, Issue 8. 2007. Pages 473-478. https://doi.org/10.1016/j.crma.2007.09.019.
2. Quarteroni, Alfio, Andrea Manzoni, and Federico Negri. Reduced Basis Methods for Partial Differential Equations. Vol. 92. UNITEXT. Cham: Springer International Publishing, 2016. http://link.springer.com/10.1007/978-3-319-15431-2.
3. Yanlai Chen, Jiang Jiahua, and Akil Narayan. A robust error estimator and a residual-free error indicator for reduced basis methods. Computers & Mathematics with Applications. 2019. http://www.sciencedirect.com/science/article/pii/S0898122118306850
4. Patrick Kürschner and Peter Benner. Efficient low-rank solutions of large-scale matrix equations. Forschungsberichte aus dem Max-Planck-Institut für Dynamik Komplexer Technischer Systeme. 2016. https://pure.mpg.de/rest/items/item_2246796_7/component/file_2296741/content.
5. Thilo Penzl. Algorithms for model reduction of large dynamical systems. Linear Algebra and its Applications. Volume 415, Issue 2, Pages 322-343. June 1, 2006. https://www.sciencedirect.com/science/article/pii/S0024379506000371.
6. D.B.P. Huynh, D.J. Knezevic, Y. Chen, J.S. Hesthaven, A.T. Patera. A natural-norm Successive Constraint Method for inf-sup lower bounds. Computer Methods in Applied Mechanics and Engineering. Volume 199, Issue 29. June 1, 2010. https://www.sciencedirect.com/science/article/pii/S0045782510000691.