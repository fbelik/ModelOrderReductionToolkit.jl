using Documenter
using MOR

makedocs(
    sitename = "MOR.jl",
    # modules  = [MOR], # If do this, must add docs for each method
    pages    = [
        "index.md" # Add additional pages here
    ]
)