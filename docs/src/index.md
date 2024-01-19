# MOR.jl Documentation

### Projections functionality: 
```@docs
singular_values_information
pca_projector
qr_projector
MOR.full_lu
eim_projector
```

### Successive Constraint Method:
```@docs
MOR.SCM_Init
initialize_SCM_SPD
initialize_SCM_Noncoercive
find_sigma_bounds
MOR.form_upperbound_set!
MOR.solve_LBs_LP
```

### Radial Basis Interpolatory Stability Factor
```@docs
min_sigma_rbf
```