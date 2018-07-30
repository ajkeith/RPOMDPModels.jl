# Crying baby problem described in DMU book
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

mutable struct BabyRPOMDP <: RPOMDP{Bool, Bool, Bool}
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry_l::Float64
    p_become_hungry_u::Float64
    p_cry_when_hungry_l::Float64
    p_cry_when_hungry_u::Float64
    p_cry_when_not_hungry_l::Float64
    p_cry_when_not_hungry_u::Float64
    discount::Float64
end

function BabyRPOMDP(r_feed, r_hungry, p_become_hungry, p_cry_when_hungry, p_cry_when_not_hungry, discount)
    p1l = p_become_hungry - p_become_hungry * 0.1
    p1u = p_become_hungry + (1 - p_become_hungry) * 0.1
    p2l = p_cry_when_hungry - p_cry_when_hungry * 0.1
    p2u = p_cry_when_hungry + (1 - p_cry_when_hungry) * 0.1
    p3l = p_cry_when_not_hungry - p_cry_when_not_hungry * 0.1
    p3u = p_cry_when_not_hungry + (1 - p_cry_when_not_hungry) * 0.1
    BabyRPOMDP(r_feed, r_hungry, p1l, p1u, p2l, p2u, p3l, p3u, discount)
end
BabyRPOMDP(r_feed, r_hungry) = BabyRPOMDP(r_feed, r_hungry, 0.1, 0.8, 0.1, 0.9)
BabyRPOMDP() = BabyRPOMDP(-5., -10.)

# updater(problem::BabyRPOMDP) = DiscreteUpdater(problem)

# start knowing baby is not not hungry
initial_state_distribution(::BabyRPOMDP) = BoolDistribution(0.0)

n_states(::BabyRPOMDP) = 2
state_index(::BabyRPOMDP, s::Bool) = s ? 1 : 2
action_index(::BabyRPOMDP, a::Bool) = a ? 1 : 2
n_actions(::BabyRPOMDP) = 2
n_observations(::BabyRPOMDP) = 2

function transition(rpomdp::BabyRPOMDP, s::Bool, a::Bool, ttype::String)
    if !a && s # did not feed when hungry
        return BoolDistribution(1.0)
    elseif a # fed
        return BoolDistribution(0.0)
    elseif ttype == "lower" # did not feed when not hungry, lower
        return BoolDistribution(rpomdp.p_become_hungry_l)
    elseif ttype == "upper" # did not feed when not hungry, lower
        return BoolDistribution(rpomdp.p_become_hungry_u)
    else
        return error("transition string input should be 'lower' or 'upper'")
    end
end

function observation(rpomdp::BabyRPOMDP, a::Bool, sp::Bool, otype::String)
    if otype == "lower"
        if sp # hungry
            return BoolDistribution(rpomdp.p_cry_when_hungry_l)
        else
            return BoolDistribution(rpomdp.p_cry_when_not_hungry_l)
        end
    elseif otype == "upper"
        if sp # hungry
            return BoolDistribution(rpomdp.p_cry_when_hungry_u)
        else
            return BoolDistribution(rpomdp.p_cry_when_not_hungry_u)
        end
    else
        return error("observation string input should be 'lower' or 'upper'")
    end
end
observation(rpomdp::BabyRPOMDP, s::Bool, a::Bool, sp::Bool, otype::String) = observation(rpomdp, a, sp, otype)

function reward(rpomdp::BabyRPOMDP, s::Bool, a::Bool)
    r = 0.0
    if s # hungry
        r += rpomdp.r_hungry
    end
    if a # feed
        r += rpomdp.r_feed
    end
    return r
end

discount(p::BabyRPOMDP) = p.discount

function generate_o(p::BabyRPOMDP, s::Bool, rng::AbstractRNG, otype::String)
    if otype == "lower"
        d = observation(p, true, s, otype) # obs distrubtion not action dependant
    elseif otype == "upper"
        d = observation(p,true, s, otype) # obs distribution not action dependant
    else
        return error("observation string input should be 'lower' or 'upper'")
    end
    return rand(rng, d)
end

# some example policies
mutable struct Starve <: Policy end
action{B}(::Starve, ::B) = false
updater(::Starve) = VoidUpdater()

mutable struct AlwaysFeed <: Policy end
action{B}(::AlwaysFeed, ::B) = true
updater(::AlwaysFeed) = VoidUpdater()

# feed when the previous observation was crying - this is nearly optimal
mutable struct FeedWhenCrying <: Policy end
updater(::FeedWhenCrying) = PreviousObservationUpdater{Bool}()
function action(::FeedWhenCrying, b::Nullable{Bool})
    if get(b, false) == false # not crying (or null)
        return false
    else # is crying
        return true
    end
end
action(::FeedWhenCrying, b::Bool) = b
action(p::FeedWhenCrying, b::Any) = action(p, initialize_belief(updater(p), b))
