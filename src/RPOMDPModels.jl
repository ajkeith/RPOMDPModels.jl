module RPOMDPModels

using RPOMDPs
using RPOMDPToolbox
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
import RPOMDPs: observation

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

include("Baby3RrhoPOMDP.jl")
export
    Baby3RrhoPOMDP
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
    observation,
    reward,
    transition

end # module
