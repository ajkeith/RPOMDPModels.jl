# Based on crying baby problem (Kochenderfer 2015)
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

mutable struct Baby3RPOMDP <: RPOMDP{Bool, Bool, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
end

mutable struct PBox
    lower::Distribution
    upper::Distribution
end
PBox(d::Distribution) = PBox(d,d) # degenerate PBox

Baby3RPOMDP(r_feed, r_hungry) = Baby3RPOMDP(r_feed, r_hungry, 0.9)
Baby3RPOMDP() = Baby3RPOMDP(-5.0, -10.0, 0.9)

# updater(problem::Baby3RPOMDP) = DiscreteUpdater(problem)

initial_state_distribution(::Baby3RPOMDP) = BoolDistribution(0.0)

const states_const = [true, false] # hungry, full
const actions_const = [true, false] # feed, no action
const observations_const = [:quiet, :crying, :yelling]

observations(::Baby3RPOMDP) = observations_const

n_states(::Baby3RPOMDP) = 2
state_index(::Baby3RPOMDP, s::Bool) = s ? 1 : 2
action_index(::Baby3RPOMDP, a::Bool) = a ? 1 : 2
observation_index(::Baby3RPOMDP, z::Symbol) = z == :quiet ? 1 : z == :crying ? 2 : 3
obs_index(p::Baby3RPOMDP, z::Symbol) = observation_index(p::Baby3RPOMDP, z::Symbol)
n_actions(::Baby3RPOMDP) = 2
n_observations(::Baby3RPOMDP) = 3

# transition data
const t_array = [BoolDistribution(0.0) BoolDistribution(1.0);
    BoolDistribution(0.0) BoolDistribution(0.1)]

function transition(rpomdp::Baby3RPOMDP, s::Bool, a::Bool)
    return t_array[state_index(rpomdp, s), action_index(rpomdp, a)]
end

# pbox representation of obesrvation uncertainty
const o_pbox = [PBox(Categorical([0.15, 0.55, 0.3]), Categorical([0.35, 0.6, 0.05])) PBox(Categorical([0.8, 0.1, 0.1]));
    PBox(Categorical([0.15, 0.55, 0.3]), Categorical([0.35, 0.6, 0.05])) PBox(Categorical([0.8, 0.1, 0.1]))]

# returns pbox representation
function observation(rpomdp::Baby3RPOMDP, a::Bool, sp::Bool)
    o_pbox[action_index(rpomdp, a), state_index(rpomdp, sp)]::PBox
end
observation(rpomdp::Baby3RPOMDP, s::Bool, a::Bool, sp::Bool) = observation(rpomdp, a, sp)

# lower and upper probability representation of the pbox
function pzinterval(rpomdp::Baby3RPOMDP)
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

function pzainterval(rpomdp::Baby3RPOMDP, a)
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


function reward(rpomdp::Baby3RPOMDP, s::Bool, a::Bool)
    r = 0.0
    if s # hungry
        r += rpomdp.r_hungry
    end
    if a # feed
        r += rpomdp.r_feed
    end
    return r
end

discount(p::Baby3RPOMDP) = p.discount

function generate_o(p::Baby3RPOMDP, s::Bool, rng::AbstractRNG, otype::String)
    d = observation(p, true, s, otype) # obs distribution not action dependant
    return observations[rand(rng, d)]
end
