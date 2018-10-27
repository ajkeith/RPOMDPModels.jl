module RPOMDPModels

using RPOMDPs
using RPOMDPToolbox
using Distributions, SimpleProbabilitySets
using JuMP, Clp
using StaticArrays
using AutoHashEquals
using StatsBase
using RCall

importall RPOMDPs

import Base.rand!
import Base.rand
import Base.==
import Base.hash

import RPOMDPs: initial_state, generate_s, generate_o, generate_sor
import RPOMDPs: observation

include("Baby2POMDP.jl")
export
    Baby2POMDP,
    Baby2RPOMDP,
    Baby2IPOMDP,
    Baby2RIPOMDP

include("BabyRobust.jl")
export
    BabyPOMDP,
    BabyRPOMDP,
    BabyIPOMDP,
    BabyRIPOMDP

include("BabyRobustInfo.jl")
export
    BabyInfoPOMDP,
    BabyInfoRPOMDP

include("SimpleBaby2Robust.jl")
export
    SimpleBaby2RPOMDP

include("SimpleTigerRobust.jl")
export
    SimpleTigerRPOMDP

include("TigerRobust.jl")
export
    TigerPOMDP,
    TigerRPOMDP,
    TigerIPOMDP,
    TigerRIPOMDP

include("TigerRobustInfo.jl")
export
    TigerInfoPOMDP,
    TigerInfoRPOMDP

include("CyberRobust.jl")
export
    CyberPOMDP,
    CyberRPOMDP,
    CyberIPOMDP,
    CyberRIPOMDP

include("CyberRobustMisspec.jl")
export
    CyberTestIPOMDP

include("SimpleRobust.jl")
export
    SimpleIPOMDP,
    SimpleRIPOMDP

include("RockDiagnosis.jl")
export
    RockIPOMDP,
    RockRIPOMDP

export
    n_states,
    n_actions,
    n_observations,
    state_index,
    action_index,
    observation_index,
    observation,
    dynamics,
    reward,
    rewardalpha,
    transition,
    observation,
    action,
    initial_belief,
    generate_sor,
    generate_sor_worst,
    initial_state_distribution,
    initial_belief_distribution

end # module
