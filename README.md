# RPOMDPModels.jl

A collection of robust, partially observable Markov decision problem formulations for extended versions of well known POMDP problems, including belief-reward variants.

## Installation
This application is built for Julia 0.6. If not already installed, the application can be cloned using

```julia
Pkg.clone("https://github.com/ajkeith/RPOMDPModels.jl")
```

## Usage

Use default problem implmentations or adjust parameters such as discount rate and ambiguity size.

```julia
using RPOMDPModels

problem = TigerRPOMDP()
states(problem)
observation(problem, :listen, :tigerleft)
```
To solve these robust POMDP models, see [RobustValueIteration](https://github.com/ajkeith/RobustValueIteration).

## Data

The cyber assessment robust belief-reward POMDP formulation is significantly larger than the other formulations. The numeric values for the two dimensional slices of the lower and upper transition and observation arrays are included in the data folder as .csv files for reproducibility. These files are produced by the `save_dynamics.jl` file.

## References
The robust POMDP model environment is a direct extension of [POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl) to the robust setting.

If this code is useful to you, please star this package and consider citing the following paper.

Egorov, M., Sunberg, Z. N., Balaban, E., Wheeler, T. A., Gupta, J. K., & Kochenderfer, M. J. (2017). POMDPs.jl: A framework for sequential decision making under uncertainty. Journal of Machine Learning Research, 18(26), 1–5.
