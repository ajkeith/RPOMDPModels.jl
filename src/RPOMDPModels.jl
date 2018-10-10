module RPOMDPModels

using RPOMDPs
using RPOMDPToolbox
using Distributions, SimpleProbabilitySets
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
    initial_state_distribution

end # module
