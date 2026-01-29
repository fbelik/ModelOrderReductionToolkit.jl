# Additional ModelOrderReductionToolkit.jl Docstrings

### Models
```@docs
galerkin_project
galerkin_add!
PoissonModel
PenzlModel
MISOPenzlModel
ParameterizedPenzlModel
to_frequency_domain
to_ss
to_dss
to_ode_problem
bode
poles
poles_and_vectors
observability_gramian
reachability_gramian
H2_norm
```

### Reductors
```@docs
StabilityResidualErrorEstimator
form_rom
lift
add_to_rb!
get_rom
```

### Affinely parameter-dependent arrays:
```@docs
APArray
formArray!
eim
```

### Matrices as vector of vectors:
```@docs
VOV
addRow!
removeRow!
addCol!
removeCol!
```

### Successive constraint method (SCM):
```@docs
SCM
ANLSCM
NNSCM
copy_scm
constrain!
```

### Radial-basis interpolatory stability factor
```@docs
ModelOrderReductionToolkit.Sigma_Min_RBF
min_sigma_rbf
update_sigma_rbf!
```

### Computation of norm of residual
```@docs
ModelOrderReductionToolkit.ResidualNormComputer
ModelOrderReductionToolkit.StandardResidualNormComputer
ModelOrderReductionToolkit.ProjectionResidualNormComputer
```

### Linear algebra utilities
```@docs
ModelOrderReductionToolkit.full_lu
ModelOrderReductionToolkit.reig
ModelOrderReductionToolkit.smallest_sval
ModelOrderReductionToolkit.largest_sval
ModelOrderReductionToolkit.orthonormalize_mgs2!
```