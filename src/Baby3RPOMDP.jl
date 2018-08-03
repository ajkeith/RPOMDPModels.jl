
# Crying baby problem described in DMU book
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

mutable struct Baby3RPOMDP <: RPOMDP{Bool, Bool, Bool}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
end

Baby3RPOMDP(r_feed, r_hungry) = Baby3RPOMDP(r_feed, r_hungry, 0.9)
Baby3RPOMDP() = Baby3RPOMDP(-5.0, -10.0, 0.9)

# updater(problem::Baby3RPOMDP) = DiscreteUpdater(problem)

# start knowing baby is not not hungry
initial_state_distribution(::Baby3RPOMDP) = BoolDistribution(0.0)

const states_const = [true, false] # hungry, full
const actions_const = [true, false] # feed, no action
const observations_const = [:quiet, :crying, :yelling]

n_states(::Baby3RPOMDP) = 2
state_index(::Baby3RPOMDP, s::Bool) = s ? 1 : 2
action_index(::Baby3RPOMDP, a::Bool) = a ? 1 : 2
observation_index(::Baby3RPOMDP, z::Symbol) = z == :quiet ? 1 : z == :crying ? 2 : 3
n_actions(::Baby3RPOMDP) = 2
n_observations(::Baby3RPOMDP) = 3

# transition data
const t_array = [BoolDistribution(0.0) BoolDistribution(1.0);
    BoolDistribution(0.0) BoolDistribution(0.1)]

function transition(rpomdp::Baby3RPOMDP, s::Bool, a::Bool)
    return t_array[state_index(rpomdp, s), action_index(rpomdp, a)]
end

# observation data
const o_pbox = [[Categorical([0.15, 0.55, 0.3]) Categorical([0.8, 0.1, 0.1]);
    Categorical([0.15, 0.55, 0.3]) Categorical([0.8, 0.1, 0.1])],
    [Categorical([0.35, 0.6, 0.05]) Categorical([0.8, 0.1, 0.1]);
        Categorical([0.35, 0.6, 0.05]) Categorical([0.8, 0.1, 0.1])]]

function observation(rpomdp::Baby3RPOMDP, a::Bool, sp::Bool, otype::String)
    if otype == "lower"
        return o_pbox[1][action_index(rpomdp, a), state_index(rpomdp, sp)]
    elseif otype == "upper"
        return o_pbox[2][action_index(rpomdp, a), state_index(rpomdp, sp)]
    else
        return error("observation string input should be 'lower' or 'upper'")
    end
end
observation(rpomdp::Baby3RPOMDP, s::Bool, a::Bool, sp::Bool, otype::String) = observation(rpomdp, a, sp, otype)

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
