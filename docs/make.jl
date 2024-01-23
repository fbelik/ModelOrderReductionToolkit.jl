using Documenter
using MOR

makedocs(
    sitename = "MOR.jl",
    modules  = [MOR],
    pages    = [
        "index.md",
        "test_prob.md"
    ],
    format = Documenter.HTML(prettyurls = false)
)