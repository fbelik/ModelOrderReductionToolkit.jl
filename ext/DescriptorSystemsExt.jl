module DescriptorSystemsExt

using ModelOrderReductionToolkit
using DescriptorSystems: dss, AbstractDescriptorStateSpace

function ModelOrderReductionToolkit.LTIModel(lti::AbstractDescriptorStateSpace)
    return LTIModel(lti.A, lti.B, lti.C, lti.D, lti.E)
end

function ModelOrderReductionToolkit.to_dss(model::LTIModel, p=nothing)
    if !isnothing(p)
        model(p)
    end
    return dss(Matrix(model.A), isa(model.E, UniformScaling) ? model.E : Matrix(model.E), Matrix(model.B), Matrix(model.C), Matrix(model.D))
end

end