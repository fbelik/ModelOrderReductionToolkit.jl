# ModelOrderReductionToolkit.jl Docstrings

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

### Projections functionality: 
```@docs
singular_values_information
pca_projector
qr_projector
eim_projector
```

### Greedy linear affine reduced basis
```@docs
ModelOrderReductionToolkit.Greedy_RB_Affine_Linear
GreedyRBAffineLinear
ModelOrderReductionToolkit.greedy_rb_err_data
ModelOrderReductionToolkit.append_affine_rbm!
ModelOrderReductionToolkit.init_affine_rbm
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
ModelOrderReductionToolkit.Affine_Residual_Init_Proj
ModelOrderReductionToolkit.Affine_Residual_Init
residual_norm_affine_proj_init
residual_norm_affine_init
add_col_to_V!
residual_norm_affine_online
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