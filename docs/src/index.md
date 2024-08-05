# ModelOrderReductionToolkit.jl Docstrings

### Models
```@docs
LinearModel
PoissonModel
LinearMatrixModel
LTIModel
to_frequency_domain
PenzlModel
MISOPenzlModel
ParameterizedPenzlModel
```

### Reductors
```@docs
PODReductor
SGReductor
WGReductor
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
ModelOrderReductionToolkit.SCM_Init
initialize_SCM_SPD
initialize_SCM_Noncoercive
find_sigma_bounds
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
ModelOrderReductionToolkit.smallest_real_eigval
ModelOrderReductionToolkit.largest_real_eigval
ModelOrderReductionToolkit.smallest_real_pos_eigpair
ModelOrderReductionToolkit.smallest_sval
ModelOrderReductionToolkit.orthonormalize_mgs2!
```