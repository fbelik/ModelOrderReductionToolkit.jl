using Documenter
using ModelOrderReductionToolkit

makedocs(
    sitename = "ModelOrderReductionToolkit.jl",
    modules  = [ModelOrderReductionToolkit],
    pages    = [
        "Docstrings" => "index.md",
        "RBM Tutorial" => "rbm_tutorial.md"
    ],
    checkdocs = :none,
    format = Documenter.HTML(prettyurls = false)
)

deploydocs(;
    repo="github.com/fbelik/ModelOrderReductionToolkit.jl.git",
)