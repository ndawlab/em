using Pkg

function setup(pkgs = Pkg.installed())
    required_pkgs = [
        "Distributed", "DataFrames", "ForwardDiff", "PyCall",
        "Distributions", "GLM", "SharedArrays"
    ];
    to_install = [];

    for (idx, val) in enumerate(required_pkgs)
        if !(val in keys(pkgs))
            Pkg.add(val)
            push!(to_install, val)
        end
    end

    to_use
end
