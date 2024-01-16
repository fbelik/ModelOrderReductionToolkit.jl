module MOR
## REMOVE REVISE AND PROGRESSBARS AND PLOTS
using LinearAlgebra
include("projections.jl")
include("successive_constraint.jl")
include("stability_radial_basis.jl")

# Linear Algebra exports
export svd
export qr
export eigen
# projections.jl exports
export singular_values_information
export pca_projector
export qr_projector
export eim_projector
# successive_constraint_spd.jl exports
export initialize_SCM_SPD
export initialize_SCM_Noncoercive
export find_sigma_bounds
# stability_radial_basis.jl exports
export min_sigma_rbf

end
