using Documenter
using ModelOrderReductionToolkit

makedocs(
    sitename = "ModelOrderReductionToolkit.jl",
    modules  = [ModelOrderReductionToolkit],
    pages    = [
        "Docstrings" => "index.md",
        "Test Problem" => "test_prob.md",
        "RBM Tutorial" => "rbm_tutorial.md"
    ],
    format = Documenter.HTML(prettyurls = false)
)

deploydocs(;
    repo="github.com/fbelik/ModelOrderReductionToolkit.jl.git",
)