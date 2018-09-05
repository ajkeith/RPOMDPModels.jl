
# Crying baby problem described in DMU book
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

mutable struct Baby3RrhoPOMDP <: RrhoPOMDP{Bool, Bool, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
end

mutable struct PBox
    lower::Distribution
    upper::Distribution
end
PBox(d::Distribution) = PBox(d,d) # degenerate PBox

Baby3RrhoPOMDP(r_feed, r_hungry) = Baby3RrhoPOMDP(r_feed, r_hungry, 0.9)
Baby3RrhoPOMDP() = Baby3RrhoPOMDP(-5.0, -10.0, 0.9)

# updater(problem::Baby3RrhoPOMDP) = DiscreteUpdater(problem)

# start knowing baby is not not hungry
initial_state_distribution(::Baby3RrhoPOMDP) = BoolDistribution(0.0)

const states_const = [true, false] # hungry, full
const actions_const = [true, false] # feed, no action
const observations_const = [:quiet, :crying, :yelling]

observations(::Baby3RrhoPOMDP) = observations_const

n_states(::Baby3RrhoPOMDP) = 2
state_index(::Baby3RrhoPOMDP, s::Bool) = s ? 1 : 2
action_index(::Baby3RrhoPOMDP, a::Bool) = a ? 1 : 2
observation_index(::Baby3RrhoPOMDP, z::Symbol) = z == :quiet ? 1 : z == :crying ? 2 : 3
obs_index(p::Baby3RrhoPOMDP, z::Symbol) = observation_index(p::Baby3RrhoPOMDP, z::Symbol)
n_actions(::Baby3RrhoPOMDP) = 2
n_observations(::Baby3RrhoPOMDP) = 3

# transition data
const t_array = [BoolDistribution(0.0) BoolDistribution(1.0);
    BoolDistribution(0.0) BoolDistribution(0.1)]

function transition(rpomdp::Baby3RrhoPOMDP, s::Bool, a::Bool)
    return t_array[state_index(rpomdp, s), action_index(rpomdp, a)]
end

# # observation data
# const o_pbox = [[Categorical([0.15, 0.55, 0.3]) Categorical([0.8, 0.1, 0.1]);
#     Categorical([0.15, 0.55, 0.3]) Categorical([0.8, 0.1, 0.1])],
#     [Categorical([0.35, 0.6, 0.05]) Categorical([0.8, 0.1, 0.1]);
#         Categorical([0.35, 0.6, 0.05]) Categorical([0.8, 0.1, 0.1])]]

# pbox representation of obesrvation uncertainty
const o_pbox = [PBox(Categorical([0.15, 0.55, 0.3]), Categorical([0.35, 0.6, 0.05])) PBox(Categorical([0.8, 0.1, 0.1]));
    PBox(Categorical([0.15, 0.55, 0.3]), Categorical([0.35, 0.6, 0.05])) PBox(Categorical([0.8, 0.1, 0.1]))]

# returns pbox representation
function observation(rpomdp::Baby3RrhoPOMDP, a::Bool, sp::Bool)
    o_pbox[action_index(rpomdp, a), state_index(rpomdp, sp)]::PBox
end
observation(rpomdp::Baby3RrhoPOMDP, s::Bool, a::Bool, sp::Bool) = observation(rpomdp, a, sp)

# lower and upper probability representation of the pbox
function pzinterval(rpomdp::Baby3RrhoPOMDP)
    nz = n_observations(rpomdp)
    na = n_actions(rpomdp)
    ns = n_states(rpomdp)
    pzl = Array{Float64}(nz,na,ns)
    pzu = Array{Float64}(nz,na,ns)
    for (aind, a) in enumerate(RPOMDPModels.ordered_actions(rpomdp)), (tind, t) in enumerate(RPOMDPModels.ordered_states(rpomdp))
        uncset = RPOMDPs.observation(rpomdp, a, t)
        cdfl = cumsum(pdf.(uncset.lower, support(uncset.lower)))
        cdfu = cumsum(pdf.(uncset.upper, support(uncset.upper)))
        pzl[1,aind,tind] = cdfl[1]
        pzu[1,aind,tind] = cdfu[1]
        for zind = 2:nz
            pzl[zind,aind,tind] = cdfl[zind] - cdfu[zind - 1]
            pzu[zind,aind,tind] = cdfu[zind] - cdfl[zind - 1]
        end
    end
    pzl, pzu
end

function pzainterval(rpomdp::Baby3RrhoPOMDP, a)
    aind = action_index(rpomdp, a)
    nz = n_observations(rpomdp)
    ns = n_states(rpomdp)
    pzl = Array{Float64}(ns,nz)
    pzu = Array{Float64}(ns,nz)
    for (tind, t) in enumerate(RPOMDPModels.ordered_states(rpomdp))
        uncset = RPOMDPs.observation(rpomdp, a, t)
        cdfl = cumsum(pdf.(uncset.lower, support(uncset.lower)))
        cdfu = cumsum(pdf.(uncset.upper, support(uncset.upper)))
        pzl[tind,1] = cdfl[1]
        pzu[tind,1] = cdfu[1]
        for zind = 2:nz
            pzl[tind,zind] = cdfl[zind] - cdfu[zind - 1]
            pzu[tind,zind] = cdfu[zind] - cdfl[zind - 1]
        end
    end
    pzl, pzu
end

# need to move this and other px functions to RPOMDPs or RPOMDPToolbox
function pinterval(rpomdp::RPOMDP)
    ns = n_states(rpomdp)
    nz = n_observations(rpomdp)
    na = n_actions(rpomdp)
    p = Array{Float64}(ns,nz,ns,na)
    plower = Array{Float64}(ns,nz,ns,na)
    pupper = Array{Float64}(ns,nz,ns,na)
    for tind in 1:ns, zind in 1:nz, sind in 1:ns, aind in 1:na
        t = RPOMDPModels.ordered_states(rpomdp)[tind]
        s = RPOMDPModels.ordered_states(rpomdp)[sind]
        a = RPOMDPModels.ordered_actions(rpomdp)[aind]
        z = RPOMDPModels.ordered_observations(rpomdp)[zind]
        dt = RPOMDPModels.transition(rpomdp, s, a)
        pt = pdf(dt, t)
        pzl, pzu = RPOMDPModels.pzainterval(rpomdp, a)
        plower[tind, zind, sind, aind] = pzl[tind, zind] * pt
        pupper[tind, zind, sind, aind] = pzu[tind, zind] * pt
    end
    plower, pupper
end

const ra1 = [1.0, -1.0] # belief-reward alpha vector
const ra2 = [-1.0, 1.0] # belief-reward alpha vector
const ra3 = [0.2, 0.2] # belief-reward alpha vector
const ralphas = [ra1, ra2, ra3]

function reward(rpomdp::Baby3RrhoPOMDP, b::Vector{Float64}, a::Bool)
    rmax = -Inf
    for α in ralphas
        rmax = max(rmax, dot(α, b))
    end
    rmax
end

function rewardalpha(rpomdp::Baby3RrhoPOMDP, b::Vector{Float64}, a::Bool)
    rmax = -Inf
    ralpha = nothing
    for α in ralphas
        rnext = dot(α, b)
        if rnext > rmax
            rmax = rnext
            ralpha = copy(α)
        end
    end
    ralpha
end

discount(p::Baby3RrhoPOMDP) = p.discount

function generate_o(p::Baby3RrhoPOMDP, s::Bool, rng::AbstractRNG, otype::String)
    d = observation(p, true, s, otype) # obs distribution not action dependant
    return observations[rand(rng, d)]
end

# # some example policies
# mutable struct Starve <: Policy end
# action{B}(::Starve, ::B) = false
# updater(::Starve) = VoidUpdater()
#
# mutable struct AlwaysFeed <: Policy end
# action{B}(::AlwaysFeed, ::B) = true
# updater(::AlwaysFeed) = VoidUpdater()
#
# # feed when the previous observation was crying - this is nearly optimal
# mutable struct FeedWhenCrying <: Policy end
# updater(::FeedWhenCrying) = PreviousObservationUpdater{Bool}()
# function action(::FeedWhenCrying, b::Nullable{Bool})
#     if get(b, false) == false # not crying (or null)
#         return false
#     else # is crying
#         return true
#     end
# end
# action(::FeedWhenCrying, b::Bool) = b
# action(p::FeedWhenCrying, b::Any) = action(p, initialize_belief(updater(p), b))
