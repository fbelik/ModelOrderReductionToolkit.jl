using Documenter
using ModelOrderReductionToolkit

makedocs(
    sitename = "ModelOrderReductionToolkit.jl",
    modules  = [ModelOrderReductionToolkit],
    pages    = [
        "Models and Reductors" => "index.md",
        "RBM Tutorial" => "rbm_tutorial.md",
        "Additional Docstrings" => "docs.md"
    ],
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls = false)
)

deploydocs(;
    repo="github.com/fbelik/ModelOrderReductionToolkit.jl.git",
)