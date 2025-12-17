module ControlSystemsExt

using ModelOrderReductionToolkit
using ControlSystems: ss, AbstractStateSpace

function ModelOrderReductionToolkit.LTIModel(lti::AbstractStateSpace)
    return LTIModel(lti.A, lti.B, lti.C, lti.D)
end

function ModelOrderReductionToolkit.to_ss(model::LTIModel, p=nothing)
    if !isnothing(p)
        model(p)
    end
    return ss(model.A, model.B, model.C, model.D)
end

end