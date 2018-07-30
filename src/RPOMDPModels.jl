module RPOMDPModels

using RPOMDPs
using POMDPToolbox
using Distributions
using StaticArrays
using AutoHashEquals
using StatsBase

importall RPOMDPs

import Base.rand!
import Base.rand
import Base.==
import Base.hash

import RPOMDPs: initial_state, generate_s, generate_o, generate_sor

include("BabyRPOMDP.jl")
export
    BabyRPOMDP,
    BabyBeliefUpdater,
    Starve,
    AlwaysFeed,
    FeedWhenCrying

include("Baby3RPOMDP.jl")
export
    Baby3RPOMDP,
    Baby3BeliefUpdater
    # Starve,
    # AlwaysFeed,
    # FeedWhenCrying

export
    n_states,
    n_actions,
    n_observations,
    state_index,
    action_index,
    observation_index,
    states,
    actions,
    observations,
    observation,
    reward,
    transition

end # module
