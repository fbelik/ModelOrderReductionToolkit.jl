module MOR

using LinearAlgebra
include("projections.jl")

# Linear Algebra exports
export svd
export qr
# projections.jl exports
export singular_values_information
export pca_projector
export qr_projector
export eim_projector

end
