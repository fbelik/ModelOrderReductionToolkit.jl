module ModelOrderReductionToolkit
using Arpack
using ControlSystems: ss, AbstractStateSpace
using DescriptorSystems: dss, AbstractDescriptorStateSpace
using JuMP
using LinearAlgebra
using NearestNeighbors
using OrdinaryDiffEq: ODEProblem
using Printf
using ProgressBars: ProgressBar
using Random: randperm
using SparseArrays
using StaticArrays
using Tulip
using UpdatableQRFactorizations
include("la_utils.jl")
include("ap_arrays.jl")
include("vector_of_vectors.jl")
include("successive_constraint.jl")
include("stability_radial_basis.jl")
include("residual_norm.jl")
include("models/model.jl")
include("models/linear_model.jl")
include("models/linear_matrix_model.jl")
include("models/lti.jl")
include("models/built_in.jl")
include("reductors/pod.jl")
include("reductors/strong_greedy.jl")
include("reductors/weak_greedy.jl")
# Linear Algebra exports
export svd
export qr
export eigen
# ap_arrays.jl exports 
export APArray
export formArray!
export eim
# vector_of_vectors.jl exports
export VOV
export addRow!
export removeRow!
export addCol!
export removeCol!
# successive_constraint_spd.jl exports
export initialize_SCM_SPD
export initialize_SCM_Noncoercive
export find_sigma_bounds
# stability_radial_basis.jl exports
export Sigma_Min_RBF
export min_sigma_rbf
export update_sigma_rbf!
# residual_norm.jl exports
export residual_norm_affine_init
export residual_norm_affine_proj_init
export residual_norm_affine_online
export residual_norm_explicit
export add_col_to_V!
# models exports
export LinearModel
export LinearMatrixModel
export LTIModel
export to_frequency_domain
export to_ode_problem
export to_ss
export to_dss
export PoissonModel
export PenzlModel
export MISOPenzlModel
export ParameterizedPenzlModel
# reductors exports
export PODReductor
export SGReductor
export WGReductor
export StabilityResidualErrorEstimator
export form_rom
export add_to_rb!
export get_rom
export lift
export galerkin_project
export galerkin_add!
export output_length
export output_type

end
